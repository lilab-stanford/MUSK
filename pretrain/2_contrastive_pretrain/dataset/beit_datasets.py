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
import os
import torch

from torchvision import datasets, transforms
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

from .masking_generator import MaskingGenerator
from PIL import Image
import random


# >>>>>>>>>> pathology transform should take care of 'resize' since it is sensitive to mpp >>>>>>>>>>#
class ConditionalResize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        prob = random.random()

        # raw image in 40x magnification
        if 800 < min(img.size) < 1050:
            if prob < 0.33:
                return img
            elif 0.66 > prob >= 0.33:
                # make it a 20x image
                return img.resize((2 * self.size, 2 * self.size), self.interpolation)
            else:
                # make it a 10x image
                return img.resize((self.size, self.size), self.interpolation)

        # raw image in 20x magnification
        elif 400 < min(img.size) < 600:
            if prob > 0.5:
                return img
            else:
                # make it a 10x image
                return img.resize((self.size, self.size), self.interpolation)
        # raw image in 10x magnification
        else:
            return img

class DataAug4BEiT(object):
    def __init__(self,
                 input_size,
                 second_input_size,
                 num_mask_patches=75,
                 window_size=14,
                 max_mask_patches_per_block=None,
                 min_mask_patches_per_block=4
                 ):

        assert input_size == second_input_size
        self.common_transform = transforms.Compose([
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(p=0.5),
            ConditionalResize(512),  # get 0.5_mpp*512 patches
            transforms.RandomCrop(input_size)
        ])
        # <<<<<<<<<< pathology transform should take care of 'resize' since it is sensitive to mpp <<<<<<<<<<#

        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(IMAGENET_INCEPTION_MEAN),
                std=torch.tensor(IMAGENET_INCEPTION_STD))
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
        # ---- pathology transform should take care of 'resize' since it is sensitive to mpp ----#
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
