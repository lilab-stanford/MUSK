import os
import warnings
import sys
import json
from subprocess import call
from collections import defaultdict
import torch

from torch.utils.data import default_collate
from PIL import Image
from .histopathology_datasets import SkinDataset, PannukeDataset, UnitopathoDataset, \
        UnitopathoRetrievalDataset, PathMMUDataset


def build_dataset(dataset_name, root="root", transform=None, split="test", download=True, annotation_file=None,
                  language="en", task="zeroshot_classification", wds_cache_dir=None, custom_classname_file=None,
                  custom_template_file=None, **kwargs):
    """
    Main function to use in order to build a dataset instance,

    dataset_name: str
        name of the dataset
    
    root: str
        root folder where the dataset is downloaded and stored. can be shared among datasets.

    transform: torchvision transform applied to images

    split: str
        split to use, depending on the dataset can have different options.
        In general, `train` and `test` are available.
        For specific splits, please look at the corresponding dataset.
    
    annotation_file: str or None
        only for datasets with captions (used for retrieval) such as COCO
        and Flickr.
    
    custom_classname_file: str or None
        Custom classname file where keys are dataset names and values are list of classnames.

    custom_template_file: str or None
        Custom template file where keys are dataset names and values are list of prompts, or dicts
        where keys are classnames and values are class-specific prompts.

    """
    train = (split == 'train')
    # >>>>>> histopathology dataset >>>>>> #
    if dataset_name == "skin":
        assert split in ("train", "test"), f"Only `train` and `test` split available for {dataset_name}"
        ds = SkinDataset(root,
                         "./data/tiles-v2.csv",
                         transform=transform,
                         train=train,
                         val=False,
                         tumor=False)


    elif dataset_name == "pannuke":
        assert split in ("train", "test"), f"Only `train` and `test` split available for {dataset_name}"
        ds = PannukeDataset(root=root, transform=transform, train=train)

    elif dataset_name == "unitopatho":
        assert split in ("train", "test"), f"Only `train` and `test` split available for {dataset_name}"
        ds = UnitopathoDataset(root=root, transform=transform, train=train)

    elif dataset_name == "unitopatho_retrieval":
        assert split in ("train", "test"), f"Only `train` and `test` split available for {dataset_name}"
        ds = UnitopathoRetrievalDataset(root=root, transform=transform, train=train)

    # zeros-shot image-text retrieval
    elif dataset_name == "pathmmu_retrieval":
        ds = PathMMUDataset(
            root=root, 
            transform=transform, 
        )
    # <<<<<< histopathology dataset <<<<<< #

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}.")

    if not hasattr(ds, "templates"):
        ds.templates = None

    return ds


class Dummy():

    def __init__(self):
        self.classes = ["blank image", "noisy image"]

    def __getitem__(self, i):
        return torch.zeros(3, 224, 224), 0

    def __len__(self):
        return 1


def get_dataset_default_task(dataset):
    if dataset in ("flickr30k", "flickr8k", "mscoco_captions", "multilingual_mscoco_captions"):
        return "zeroshot_retrieval"
    elif dataset.startswith("sugar_crepe"):
        return "image_caption_selection"
    else:
        return "zeroshot_classification"


def get_dataset_collate_fn(dataset_name):
    if dataset_name in (
    "mscoco_captions", "multilingual_mscoco_captions", "flickr30k", "flickr8k") or dataset_name.startswith(
            "sugar_crepe"):
        return image_captions_collate_fn
    else:
        return default_collate


def has_gdown():
    return call("which gdown", shell=True) == 0


def has_kaggle():
    return call("which kaggle", shell=True) == 0

def _extract_task(dataset_name):
    prefix, *task_name_list = dataset_name.split("_")
    task = "_".join(task_name_list)
    return task


def image_captions_collate_fn(batch):
    transposed = list(zip(*batch))
    imgs = default_collate(transposed[0])
    texts = transposed[1]
    return imgs, texts


def get_dataset_collection_from_file(path):
    return [l.strip() for l in open(path).readlines()]

