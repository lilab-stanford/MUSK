# --------------------------------------------------------
# MUSK: A Vision-Language Foundation Model for Precision Oncology
# Published in Nature, 2025
# GitHub Repository: https://github.com/lilab-stanford/MUSK
# Copyright (c) 2025 Stanford University, by Jinxi Xiang
# Licensed under the CC-BY-NC-ND 4.0 License (https://creativecommons.org/licenses/by-nc-nd/4.0/)
# Please see LICENSE for additional details.
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model
import numpy as np
from typing import Optional, List, Tuple

from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from .torchscale.model.BEiT3 import BEiT3
from .torchscale.architecture.config import EncoderConfig
import math
from .utils import MultiScaleForward

class TwoLayerMLP(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features,
            out_features,
            norm_layer,
            norm_input=True,
    ):
        super().__init__()
        self.norm1 = norm_layer(in_features) if norm_input else nn.Identity()
        self.dense1 = nn.Linear(in_features, hidden_features)
        self.norm2 = norm_layer(hidden_features)
        self.act = nn.GELU()
        self.dense2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.norm1(x)
        x = self.dense1(x)
        x = self.norm2(x)
        x = self.act(x)
        return self.dense2(x)


class Pooler(nn.Module):
    def __init__(self, input_features, output_features, norm_layer):
        super().__init__()
        self.norm = norm_layer(input_features)
        self.dense = nn.Linear(input_features, output_features)
        self.activation = nn.Tanh()

    def forward(self, x):
        cls_rep = x[:, 0, :]
        cls_rep = self.norm(cls_rep)
        pooled_output = self.dense(cls_rep)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ModelWrapper(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.beit3 = BEiT3(args)
        self.apply(self._init_weights)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return self.beit3.encoder.num_layers

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token',
                'beit3.encoder.embed_positions.A.weight',
                'beit3.vision_embed.cls_token', 'logit_scale'}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            

class MUSK(ModelWrapper):
    def __init__(self, args, **kwargs):
        super().__init__(args=args)
        embed_dim = args.encoder_embed_dim

        # Define heads for vision and language
        self.language_head = nn.Linear(embed_dim, embed_dim, bias=False)
        self.vision_head = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Logit scale parameter initialization
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(
        self, 
        image: Optional[torch.Tensor] = None, 
        text_description: Optional[torch.Tensor] = None, 
        padding_mask: Optional[torch.Tensor] = None, 
        return_global: bool = True, 
        with_head: bool = True, 
        out_norm: bool = True,
        ms_aug: bool = False, 
        scales: Optional[List[int]] = None, 
        max_split_size: Optional[int] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass for vision-language model.
        Args:
            image: Input image tensor.
            text_description: Input text tokens.
            padding_mask: Padding mask for text.
            return_global: Whether to return global CLS token.
            with_head: Whether to apply linear heads.
            out_norm: Whether to normalize output embeddings.
            ms_aug: Enable multiscale feature augmentation. 
            scales: List of scales for multiscale feature augmentation.
            max_split_size: Maximum split size for multiscale forward.

        Returns:
            vision_cls: Vision embeddings (normalized if out_norm).
            language_cls: Language embeddings (normalized if out_norm).
        """
        if scales is None:
            scales = [1, 2]  # Default scales

        # Process image input
        vision_cls = None
        if image is not None:
            if ms_aug:
                vision_cls = MultiScaleForward(
                    model=self, 
                    input=image,
                    scales=scales,
                    max_split_size=max_split_size
                )
                if with_head:
                    vision_cls = self.vision_head(vision_cls[:, :1024])
            else:
                outputs = self.beit3(visual_tokens=image)
                x = outputs["encoder_out"]
                vision_cls = x[:, 0, :] if return_global else x
                if with_head:
                    vision_cls = self.vision_head(vision_cls)
            if out_norm:
                vision_cls = F.normalize(vision_cls, dim=-1)

        # Process text input
        language_cls = None
        if text_description is not None:
            outputs = self.beit3(
                textual_tokens=text_description,
                text_padding_position=padding_mask,
            )
            x = outputs["encoder_out"]
            language_cls = x[:, 0, :] if return_global else x
            if with_head:
                language_cls = self.language_head(language_cls)
            if out_norm:
                language_cls = F.normalize(language_cls, dim=-1)

        return vision_cls, language_cls


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


def _get_large_config(
        img_size=224, patch_size=16, drop_path_rate=0,
        checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs
):
    return EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True,
        layernorm_embedding=False, normalize_output=True, no_output_layer=True,
        drop_path_rate=drop_path_rate, encoder_embed_dim=1024, encoder_attention_heads=16,
        encoder_ffn_embed_dim=int(1024 * mlp_ratio), encoder_layers=24,
        checkpoint_activations=checkpoint_activations,
    )

class ModelWrapper(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.beit3 = BEiT3(args)
        self.apply(self._init_weights)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return self.beit3.encoder.num_layers

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token',
                'beit3.encoder.embed_positions.A.weight',
                'beit3.vision_embed.cls_token', 'logit_scale'}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


@register_model
def musk_large_patch16_384(pretrained=False, **kwargs):
    args = _get_large_config(img_size=384, **kwargs)
    model = MUSK(args, **kwargs)
    return model
