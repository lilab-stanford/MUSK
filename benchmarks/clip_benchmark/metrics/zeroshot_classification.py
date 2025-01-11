"""
Code adapated from https://github.com/mlfoundations/open_clip/blob/main/src/training/zero_shot.py
Thanks to the authors of OpenCLIP
"""
import logging
import pdb
from contextlib import suppress
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import glob
from sklearn.metrics import classification_report, balanced_accuracy_score


def xlm_tokenizer(tokens, tokenizer, max_len=100):
    tokens = tokenizer.encode(tokens)
    
    tokens = tokens[1:-1]  # remove eos and bos;
    if len(tokens) > max_len - 2:
        tokens = tokens[:max_len - 2]

    tokens = [tokenizer.bos_token_id] + tokens[:] + [tokenizer.eos_token_id]  # add eos and bos
    num_tokens = len(tokens)
    padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)

    text_tokens = tokens + [tokenizer.pad_token_id] * (max_len - num_tokens)
    return text_tokens, padding_mask


def zero_shot_classifier(model, tokenizer, classnames, templates, device, amp=True):
    """
    This function returns zero-shot vectors for each class in order
    to use it for zero-shot classification.
    

    model:
        CLIP-like model with `encode_text`
    
    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers
    
    classnames: list of str
        name of classes
    
    templates: list of str
        templates to use.
    
    Returns
    -------
    
    torch.Tensor of shape (N,C) where N is the number
    of templates, and C is the number of classes.
    """
    autocast = torch.cuda.amp.autocast if amp else suppress
    with torch.no_grad(), autocast():
        zeroshot_weights = []

        for classname in tqdm(classnames):
            
            if type(templates) == dict:
                # class-specific prompts (e.g., CuPL https://arxiv.org/abs/2209.03320)
                texts = templates[classname]
            elif type(templates) == list:
                # generic prompts tht are specialized for each class by replacing {c} with the class name
                texts = [template.format(c=classname) for template in templates]
            else:
                raise ValueError("templates must be a list or a dict")

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
                class_embeddings = model(
                    text_description=text_ids.to(device),
                    padding_mask=paddings.to(device),
                    with_head=True,
                    out_norm=True
                )[1]

                class_embedding = class_embeddings.mean(dim=0)
                # class_embedding /= class_embedding.norm()

            elif tokenizer.__class__.__name__ == "CLIPTokenizerFast":
                inputs = tokenizer(texts, padding=True, return_tensors="pt")
                text_features = model.get_text_features(inputs['input_ids'].to(device),
                                                        inputs['attention_mask'].to(device))

                class_embedding = text_features.mean(dim=0)
                class_embedding /= class_embedding.norm()
            
            # tokenizer for CONCH
            elif tokenizer.__class__.__name__ == "PreTrainedTokenizerFast":
                from conch.open_clip_custom import tokenize
                tokenized_prompts = tokenize(texts=texts, tokenizer=tokenizer).to(device)
                text_features = model.encode_text(tokenized_prompts)
                class_embedding = text_features.mean(dim=0)
                class_embedding /= class_embedding.norm()

            else:

                texts = tokenizer(texts).to(device)  # tokenize
                class_embeddings = model.encode_text(texts)
                class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
                class_embedding /= class_embedding.norm()

            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    """
    Compute top-k accuracy

    output: torch.Tensor
        shape (N, C) where N is the number of examples, C the number of classes.
        these are the logits.
    
    target: torch.Tensor
        shape (N,) where N is the number of examples. Groundtruth class id of each example.
    
    topk: tuple
        which topk to compute, e.g., topk=(1,5) will compute top-1 and top-5 accuracies
    
    Returns
    -------
    
    list of top-k accuracies in the same order as `topk`
    """
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    n = len(target)
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) / n for k in topk]


def run_classification(model, classifier, dataloader, device, amp=True):
    """
    Run zero-shot classifcation

    model: torch.nn.Module
        CLIP-like model with `encode_image` and `encode_text`
    
    classifier: torch.Tensor
        obtained from the function `zero_shot_classifier`
    
    dataloader: torch.utils.data.Dataloader 
    
    Returns
    -------
    (pred, true)  where
        - pred (N, C) are the logits
        - true (N,) are the actual classes
    """
    autocast = torch.cuda.amp.autocast if amp else suppress
    pred = []
    true = []
    nb = 0
    
    with torch.no_grad():
        for images, target in tqdm(dataloader):
            images = images.to(device)
            target = target.to(device)
            
            with autocast():
                model_name = model.__class__.__name__.lower()

                # predict for musk model
                if 'musk' in model_name:
                    image_features = model(
                        image=images, 
                        out_norm=True,
                        with_head=True  # head must be used for zero-shot task
                        )[0]

                elif 'clipmodel' in model_name:
                    image_features = model.get_image_features(images)
                
                # image embeddings for CONCH
                elif 'CoCa' in model_name:
                    """
                    https://github.com/mahmoodlab/CONCH
                    """
                    image_features = model.encode_image(
                        images, 
                        proj_contrast=True, 
                        normalize=True
                        )

                # predict for clip model
                else:
                    image_features = model.encode_image(images)
                    image_features = F.normalize(image_features, dim=-1)

                logits = 100. * image_features @ classifier

            true.append(target.cpu())
            pred.append(logits.float().cpu())

    pred = torch.cat(pred)
    true = torch.cat(true)
    return pred, true


def average_precision_per_class(scores, targets):
    """
    Compute average precision  for each class
    this metric is used for multi-label classification
    see explanations here https://fangdahan.medium.com/calculate-mean-average-precision-map-for-multi-label-classification-b082679d31be
    Code is adapted from https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py, thanks to the authors of `tnt`.

    Parameters
    ----------

    scores: torch.Tensor
        logits, of shape (N,C) where N is the number of examples, C the number of classes
    
    targets: torch.Tensor
        one-hot vectors of groundtruth targets (N, C), where N is the number of examples, C is the
        number of classes
    
    Returns
    -------

    torch.Tensor of shape (C,) of avereage precision for each class, where C is     
    the number of classes.
    
    """
    ap = torch.zeros(scores.size(1))
    rg = torch.arange(1, scores.size(0) + 1).float()
    # compute average precision for each class
    for k in range(scores.size(1)):
        # sort scores
        scores_k = scores[:, k]
        targets_k = targets[:, k]
        _, sortind = torch.sort(scores_k, 0, True)
        truth = targets_k[sortind]
        tp = truth.float().cumsum(0)
        # compute precision curve
        precision = tp.div(rg)
        # compute average precision
        ap[k] = precision[truth.bool()].sum() / max(float(truth.sum()), 1)
    return ap


def evaluate(model, dataloader, tokenizer, classnames, templates, device, amp=True, verbose=False, save_clf=None,
             load_clfs=[]):
    """
    Run zero-shot classification and evaluate the metrics

    Parameters
    ----------

    model: torch.nn.Module
        CLIP-like model with `encode_image` and `encode_text`
    
    dataloader: torch.utils.data.Dataloader

    tokenizer: text tokenizer

    classnames: list of str
        class names
    
    templates: list of str
        templates to use for zero-shot classification
    
    device: cpu/cuda

    amp: whether to use automatic mixed precision

    verbose: whether to use verbose model

    Returns
    -------

    dict of classification metrics
    """
    if len(load_clfs) > 0:
        n = len(load_clfs)
        classifier = torch.load(load_clfs[0], map_location='cpu') / n
        for i in range(1, n):
            classifier = classifier + torch.load(load_clfs[i], map_location='cpu') / n
        classifier = classifier.to(device)
    else:
        model = model.to(device)
        classifier = zero_shot_classifier(model, tokenizer, classnames, templates, device, amp=amp)

    if save_clf is not None:
        torch.save(classifier, save_clf)
        # exit() - not sure if we want to exit here or not.

    logits, target = run_classification(model, classifier, dataloader, device, amp=amp)
    
    is_multilabel = (len(target.shape) == 2)

    if is_multilabel:
        if verbose:
            print("Detected a multi-label classification dataset")
        # Multiple labels per image, multiple classes on the dataset
        ap_per_class = average_precision_per_class(logits, target)
        if verbose:
            for class_name, ap in zip(dataloader.dataset.classes, ap_per_class.tolist()):
                print(f"Class: {class_name}, AveragePrecision: {ap}")
        return {"mean_average_precision": ap_per_class.mean().item()}
    else:
        # Single label per image, multiple classes on the dataset
        # just compute accuracy and mean_per_class_recall

        pred = logits.argmax(axis=1)

        # measure accuracy
        if len(dataloader.dataset.classes) >= 5:
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        else:
            acc1, = accuracy(logits, target, topk=(1,))
            acc5 = float("nan")
        mean_per_class_recall = balanced_accuracy_score(target, pred)
        if verbose:
            print(classification_report(target, pred, digits=3))
        return {"balanced_acc": mean_per_class_recall}