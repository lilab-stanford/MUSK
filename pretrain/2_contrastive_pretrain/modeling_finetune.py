# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model
import numpy as np

import utils
from modeling_utils import BEiT3Wrapper, MUSKWrapper, _get_base_config, _get_large_config
from mask_decoder import AttnBlock, Attention
from torchscale.model.BEiT3 import BEiT3

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


class MUSK(BEiT3Wrapper):
    def __init__(
            self,
            args,
            **kwargs
    ):
        super(MUSK, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        self.language_head = nn.Linear(embed_dim, embed_dim, bias=False)
        self.vision_head = nn.Linear(embed_dim, embed_dim, bias=False)
        self.language_head.apply(self._init_weights)
        self.vision_head.apply(self._init_weights)

        # hard-code setting for XLMRobertaTokenizer
        self.mask_token_id = 64001
        self.vocab_size = 64010

        # create a vision-language decoder
        num_blocks = 4
        self.vl_decoder = torch.nn.ModuleList([AttnBlock(embed_dim, 16) for _ in range(num_blocks)])
        self.mlm_head = nn.Linear(embed_dim, self.vocab_size)
        self.mim_head = nn.Linear(embed_dim, 8192)

        # contrastive image-text loss
        self.criterion = utils.ClipLoss(
            rank=utils.get_rank(),
            world_size=utils.get_world_size(),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(
            self, 
            image=None, 
            text_description=None, 
            language_labels=None, 
            padding_mask=None, 
            **kwargs
            ):
        
        if image is not None:
            outputs = self.beit3(
                textual_tokens=None,
                visual_tokens=image,
                text_padding_position=None,
                vision_masked_position=None
            )
            x = outputs["encoder_out"]
            vision_cls = self.vision_head(x[:, 0, :])
            vision_states = x[:, 1:, :]
            vision_cls = F.normalize(vision_cls, dim=-1)
        else:
            vision_states = None
            vision_cls = None

        if text_description is not None:
            outputs = self.beit3(
                textual_tokens=text_description,
                visual_tokens=None,
                text_padding_position=padding_mask,
            )
            x = outputs["encoder_out"]

            language_cls = self.language_head(x[:, 0, :])
            language_states = x
            language_cls = F.normalize(language_cls, dim=-1)
        else:
            language_states = None
            language_cls = None
        
        
        if vision_cls is None or language_cls is None:
            mlm_text_loss = 0
            mlm_acc = 0
        else:
            # after text and image encoding, use decoder to predict the masked token of language
            text_feats = language_states
            for block in self.vl_decoder:
                text_feats = block(text_feats, vision_states)

            mlm_text_logits = self.mlm_head(text_feats)

            bool_mask = (text_description != self.mask_token_id).to(dtype=torch.bool, device=text_feats.device)
            mlm_text_labels = language_labels.to(device=text_feats.device)
            mlm_text_labels.masked_fill_(bool_mask, -100)

            mlm_text_loss = F.cross_entropy(
                mlm_text_logits.view(-1, self.vocab_size),
                mlm_text_labels.view(-1),
                ignore_index=-100,
            )
            # compute acc
            bool_mask = ~bool_mask
            score = torch.max(mlm_text_logits[bool_mask], -1)[1].data == language_labels[bool_mask]
            mlm_acc = torch.sum(score.float()) / torch.sum(bool_mask)
        
        if vision_cls is None or language_cls is None:
            ct_loss = 0
        else:
            ct_loss = self.criterion(vision_cls, language_cls, self.logit_scale.exp())
        
        return ct_loss, mlm_text_loss, mlm_acc



# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


@register_model
def musk_base_patch16_384(pretrained=False, **kwargs):
    args = _get_base_config(img_size=384, **kwargs)
    model = MUSK(args, **kwargs)
    return model


@register_model
def musk_large_patch16_384(pretrained=False, **kwargs):
    args = _get_large_config(img_size=384, **kwargs)
    model = MUSK(args, **kwargs)
    return model
