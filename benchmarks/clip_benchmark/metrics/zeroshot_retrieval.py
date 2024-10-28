import logging
from contextlib import suppress
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


def evaluate(model, dataloader, tokenizer,  device, amp=True, recall_k_list=[5]):
    """
    Evaluate the model on the given dataset

    Parameters
    ----------
    
    model: torch.nn,Module
        CLIP-like model with `encode_image` and `encode_text`
    
    dataloader: torch.utils.data.Dataloader
        dataloader to use for evaluation

    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers
    
    device: cpu/cuda

    amp: whether to use automatic mixed precision

    recall_k_list: list of int
        recall@k k's to use
    
    Returns
    -------
    
    dict of retrieval metrics
    """
    # list of batch of images embedding
    batch_images_emb_list = []
    # list of batch of text embedding
    batch_texts_emb_list = []
    # for each text, we collect the corresponding image index, as each image can have multiple corresponding texts
    texts_image_index = []

    dataloader = dataloader_with_indices(dataloader)
    autocast = torch.cuda.amp.autocast if amp else suppress
    
    with autocast():
        model.to(device)

    for batch_images, batch_texts, inds in tqdm(dataloader):
        batch_images = batch_images.to(device)
        
        # store the index of image for each text
        # batch_texts_image_index = [ind for ind, texts in zip(inds, batch_texts) for text in texts]

        # compute the embedding of images and texts
        with torch.no_grad(), autocast():            
            batch_texts_emb = get_text_embeddings(model, tokenizer, batch_texts, device)
            batch_images_emb = get_image_embeddings(model, batch_images)            
            

        batch_images_emb_list.append(batch_images_emb.cpu())
        batch_texts_emb_list.append(batch_texts_emb.cpu())
        # texts_image_index.extend(batch_texts_image_index)
        
        texts_image_index.extend(inds)

        
    batch_size = len(batch_images_emb_list[0])

    # concatenate all embeddings
    images_emb = torch.cat(batch_images_emb_list).float()
    texts_emb = torch.cat(batch_texts_emb_list).float()

    # get the score for each text and image pair
    scores  = texts_emb @ images_emb.t()

    # construct a the positive pair matrix, which tells whether each text-image pair is a positive or not
    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True
    metrics = {}
    for recall_k in recall_k_list:
        # Note that recall_at_k computes **actual** recall i.e. nb_true_positive/nb_positives, where the number
        # of true positives, e.g. for text retrieval, is, for each image,  the number of retrieved texts matching that image among the top-k.
        # Also, the number of positives are the total number of texts matching the image in the dataset, as we have a set of captions
        # for each image, that number will be greater than 1 for text retrieval.
        # However, image/text retrieval recall@k, the way it is done in CLIP-like papers, is a bit different.
        # recall@k, in CLIP-like papers, is, for each image, either 1 or 0. It is 1 if atleast one text matches the image among the top-k.
        # so we can easily compute that using the actual recall, by checking whether there is at least one true positive,
        # which would be the case if the recall is greater than 0. One we compute the recal for each image (or text), we average
        # it over the dataset.
        metrics[f"image_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores, positive_pairs, batch_size, device, k=recall_k)>0).float().mean().item()
        metrics[f"text_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores.T, positive_pairs.T, batch_size, device, k=recall_k)>0).float().mean().item()

    return metrics


def xlm_tokenizer(tokens, tokenizer, max_len=64):
    tokens = tokenizer.encode(tokens)

    tokens = tokens[1:-1]  # remove eos and bos;
    if len(tokens) > max_len - 2:
        tokens = tokens[:max_len - 2]
    tokens = [tokenizer.bos_token_id] + tokens[:] + [tokenizer.eos_token_id]  # ADD eos and bos;

    num_tokens = len(tokens)
    padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)

    text_tokens = tokens + [tokenizer.pad_token_id] * (max_len - num_tokens)
    return text_tokens, padding_mask


def get_text_embeddings(model, tokenizer, texts, device):
    
    # MUSK tokenizer for encoding class names
    if tokenizer.__class__.__name__ == "XLMRobertaTokenizer":

        text_ids = []
        paddings = []
        for txt in texts:
            txt_ids, pad = xlm_tokenizer(txt, tokenizer, max_len=100)
            text_ids.append(torch.tensor(txt_ids).unsqueeze(0))
            paddings.append(torch.tensor(pad).unsqueeze(0))

        text_ids = torch.cat(text_ids)
        paddings = torch.cat(paddings)
        class_embedding = model(
            text_description=text_ids.to(device),
            padding_mask=paddings.to(device),
            out_norm=True,
            with_head=True  # MUST use pretrained head for retrieval!!!!!!
        )[1]
    # transfomers clip; for PLIP
    elif tokenizer.__class__.__name__ == "CLIPTokenizerFast":
        inputs = tokenizer(
            texts, 
            padding=True, 
            truncation=True,
            max_length=77, 
            return_tensors="pt"
        )
        
        class_embedding = model.get_text_features(inputs['input_ids'].to(device),
                                                inputs['attention_mask'].to(device))

        # class_embedding = F.normalize(class_embeddings, dim=-1)
    
    # tokenizer for CONCH
    elif tokenizer.__class__.__name__ == "PreTrainedTokenizerFast":
        from conch.open_clip_custom import tokenize
        tokenized_prompts = tokenize(texts=texts, tokenizer=tokenizer).to(device)
        class_embedding = model.encode_text(tokenized_prompts)

    else:
        texts = tokenizer(texts).to(device)  # tokenize
        class_embeddings = model.encode_text(texts)
        class_embedding = F.normalize(class_embeddings, dim=-1)

    return class_embedding


def get_image_embeddings(model, batch_images):
    
    model_name = model.__class__.__name__  # quilt1m --> 'clip'; plip --> 'clipmodel';
    
    if 'musk' in model_name.lower():
        image_features = model(
            image=batch_images,
            text_description=None,
            padding_mask=None,
            out_norm=True,
            with_head=True # MUST use pretrained head for retrieval!!!!!!
        )[0]

    # CTransPath
    elif 'swin' in model_name.lower():
        image_features = model(batch_images)
        image_features = F.normalize(image_features, dim=-1)

    elif 'clipmodel' in model_name.lower():
        image_features = model.get_image_features(batch_images)
        image_features = F.normalize(image_features, dim=-1)

    # image embeddings for CONCH
    elif 'CoCa' in model_name:
        image_features = model.encode_image(batch_images, proj_contrast=True, normalize=True)

    else:
        # note: not sure if we want to train on l2-normalized features
        image_features = model.encode_image(batch_images)
        image_features = F.normalize(image_features, dim=-1)

    return image_features

def dataloader_with_indices(dataloader):
    start = 0
    for x, y in dataloader:
        
        end = start + len(x)
        inds = torch.arange(start, end)
        yield x, y, inds
        start = end

def recall_at_k(scores, positive_pairs, k):
    """
    Compute the recall at k for each sample
    :param scores: compability score between  text and image embeddings (nb texts, nb images)
    :param k: number of images to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs (nb texts, nb images)
    :return: recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # for each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # nb_texts, k, nb_images
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    # compute number of true positives
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1,2))
    # compute recall at k
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k

def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)
