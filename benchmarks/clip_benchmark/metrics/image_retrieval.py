import logging
from contextlib import suppress
import os
import glob
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .zeroshot_retrieval import batchify, dataloader_with_indices, recall_at_k
import numpy as np


def evaluate(model, dataloader, device, amp=True, recall_k_list=[1, 3, 5]):
    """
    Evaluate the model on the given dataset for image retrieval

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

    # for each text, we collect the corresponding image index, as each image can have multiple corresponding texts
    image_labels = []

    dataloader = dataloader_with_indices(dataloader)
    autocast = torch.cuda.amp.autocast if amp else suppress

    for idx, batch in tqdm(enumerate(dataloader)):
        batch_images, labels = batch[0], batch[1]
        batch_images = batch_images.to(device)
        model.to(dtype=torch.float16, device=device)

        # compute the embedding of images and texts
        with torch.no_grad(), autocast():

            model_name = model.__class__.__name__
            if 'musk' in model_name.lower():
                outputs = model(
                    image=batch_images,
                    text_description=None,
                    padding_mask=None,
                    out_norm=True,
                    with_head=True
                )
                image_features = outputs[0]

            elif 'clipmodel' in model_name.lower():
                image_features = model.get_image_features(batch_images)

            else:
                # note: not sure if we want to train on l2-normalized features
                image_features = model.encode_image(
                    batch_images,
                    proj_contrast=True, 
                    normalize=False
                    )

            batch_images_emb = image_features

        batch_images_emb_list.append(batch_images_emb.cpu())
        image_labels.append(labels)

    # concatenate all embeddings
    images_emb = torch.cat(batch_images_emb_list).float()
    image_labels = torch.cat(image_labels)
    

    metrics = image_retrieval_metrics(images_emb, image_labels)

    return metrics


def calculate_distances(batch_images):
    dot_product = torch.mm(batch_images, batch_images.t())
    square_norms = dot_product.diag()
    distances = -2 * dot_product + square_norms[:, None] + square_norms[None, :]
    distances.fill_diagonal_(float('inf'))
    return distances


def image_retrieval_metrics(batch_images, labels):
    """
    Args:
        batch_images: (batch_size, feature_dim)
        labels:  (batch_size, )
    Returns: acc@top1, 3, 5; mmv@top5
    """
    # Calculate the pairwise distances between images
    distances = torch.cdist(batch_images, batch_images)

    # Set the diagonal elements to a large value to exclude self-matching
    distances.fill_diagonal_(float('inf'))

    # # TODO: save similarity matrix for further investigations
    # save_dir = "./results/image_retrieval"
    # os.makedirs(save_dir, exist_ok=True)
    
    # sim_cnt = list(glob.glob(f"{save_dir}/sim_*"))
    # cnt = len(sim_cnt)
    # with open(f"{save_dir}/sim_{cnt}.npy", 'wb') as f:
    #     np.save(f, distances.numpy())
    
    # label_cnt = list(glob.glob(f"{save_dir}/label_*"))
    # cnt = len(label_cnt)
    # with open(f"{save_dir}/label_{cnt}.npy", 'wb') as f:
    #     np.save(f, labels.numpy())


    # Get the indices of the sorted distances
    sorted_indices = torch.argsort(distances, dim=1)

    # Get the top 1, top 3, and top 5 nearest neighbors
    top1_indices = sorted_indices[:, :1]
    top3_indices = sorted_indices[:, :3]
    top5_indices = sorted_indices[:, :5]

    # Get the labels of the top 1, top 3, and top 5 nearest neighbors
    top1_labels = labels[top1_indices].view(-1, 1)
    top3_labels = labels[top3_indices].view(-1, 3)
    top5_labels = labels[top5_indices].view(-1, 5)

    # Calculate acc@top1, acc@top3, and acc@top5
    acc_top1 = torch.mean((top1_labels == labels.view(-1, 1)).float()).item()
    acc_top3 = torch.mean((top3_labels == labels.view(-1, 1)).any(dim=1).float()).item()
    acc_top5 = torch.mean((top5_labels == labels.view(-1, 1)).any(dim=1).float()).item()

    # Calculate mMv@top5
    num_classes = len(torch.unique(labels))
    top5_label_counts = torch.zeros(top5_labels.size(0), num_classes)
    for i in range(top5_labels.size(0)):
        top5_label_counts[i] = torch.bincount(top5_labels[i], minlength=num_classes)
    majority_vote_labels = torch.argmax(top5_label_counts, dim=1)
    mmv_top5 = torch.mean((majority_vote_labels == labels).float()).item()
    return {"acc_top1": acc_top1, "acc_top3": acc_top3, "acc_top5": acc_top5, "mMv_top5": mmv_top5}
