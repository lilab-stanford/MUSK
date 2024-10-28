# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, DINO and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import torch
from torchvision import transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from PIL import Image
import random

from .masking_generator import MaskingGenerator
from .multimodal_dataset import MMDataset, Quilt1mDataset
from .randstainna import RandStainNA


class ConditionalResize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        prob = random.random()

        # raw image in 40x magnification
        if 800 < min(img.size) < 1050:
            if prob < 0.5:
                # make it a 20x image
                return img.resize((2 * 256, 2 * 256), self.interpolation)
            else:
                # make it a 40x image
                return img

        # raw image in 20x magnification
        elif 400 < min(img.size) < 600:
            if prob > 0.5:
                return img
            else:
                # make it a 10x image
                return img.resize((self.size, self.size), self.interpolation)
        # raw image in 10x magnification
        else:
            return img.resize((self.size, self.size), self.interpolation)
        

class DataAug4BEiT(object):
    def __init__(self,
                 input_size,
                 second_input_size,
                 num_mask_patches=75,
                 window_size=14,
                 max_mask_patches_per_block=None,
                 min_mask_patches_per_block=4
                 ):
        # beit1 use imagenet mean and std
        # mean = IMAGENET_DEFAULT_MEAN
        # std = IMAGENET_DEFAULT_STD

        # vlmo/beit3 use inception mean and std
        mean = IMAGENET_INCEPTION_MEAN
        std = IMAGENET_INCEPTION_STD

        assert input_size == second_input_size
        self.common_transform = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2),
            RandStainNA(
                    yaml_file = '../datasets/stain_params.yaml',
                    std_hyper = -0.3,
                    distribution = 'normal',
                    probability = 1.0,
                    is_train = True
                ),
            transforms.RandomHorizontalFlip(p=0.5),
            ConditionalResize(input_size),
            transforms.RandomCrop(input_size)
        ])
        # <<<<<<<<<< pathology transform should take care of 'resize' since it is sensitive to mpp <<<<<<<<<<#

        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        self.visual_token_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.masked_position_generator = MaskingGenerator(
            window_size, num_masking_patches=num_mask_patches,
            max_num_patches=max_mask_patches_per_block,
            min_num_patches=min_mask_patches_per_block,
        )

    def __call__(self, image):
        for_patches = self.common_transform(image.copy())
        for_visual_tokens = for_patches.copy()

        return self.patch_transform(for_patches), \
               self.visual_token_transform(for_visual_tokens), \
               self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  visual_tokens_transform = %s,\n" % str(self.visual_token_transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        if args.crop_pct is None:
            if args.input_size < 384:
                args.crop_pct = 224 / 256
            else:
                args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def build_dataset(
        input_size,
        second_input_size,
        window_size,
        max_text_len,
        image_dir, 
        text_dir, 
        tokenizer, 
        image_index, 
        text_index, 
        num_mask_patches,
        min_mask_patches_per_block,
        max_mask_patches_per_block
        ):
    
    # create image augmentations
    args_local = {"input_size": input_size,
                  "second_input_size": second_input_size,
                  "num_mask_patches": num_mask_patches,
                  "window_size": window_size,
                  "max_mask_patches_per_block": max_mask_patches_per_block,
                  "min_mask_patches_per_block": min_mask_patches_per_block}

    trans = DataAug4BEiT(**args_local)

    mmdataset = MMDataset(
        text_tokenizer=tokenizer,
        transforms=trans,
        image_dir=image_dir,
        text_dir=text_dir,
        max_length=max_text_len,
        image_index=image_index,
        text_index=text_index
    )

    return mmdataset


def build_quilt1m_dataset(
        input_size,
        second_input_size,
        window_size,
        max_text_len,
        image_dir, 
        text_dir, 
        tokenizer, 
        image_index, 
        text_index, 
        num_mask_patches,
        min_mask_patches_per_block,
        max_mask_patches_per_block
        ):
    
    # create image augmentations
    args_local = {"input_size": input_size,
                  "second_input_size": second_input_size,
                  "num_mask_patches": num_mask_patches,
                  "window_size": window_size,
                  "max_mask_patches_per_block": max_mask_patches_per_block,
                  "min_mask_patches_per_block": min_mask_patches_per_block}

    trans = DataAug4BEiT(**args_local)

    mmdataset = Quilt1mDataset(
        text_tokenizer=tokenizer,
        transforms=trans,
        image_dir=image_dir,
        text_dir=text_dir,
        max_length=max_text_len,
        image_index=image_index,
        text_index=text_index
    )

    return mmdataset
