# --------------------------------------------------------
# MUSK: A Vision-Language Foundation Model for Precision Oncology
# Published in Nature, 2025
# GitHub Repository: https://github.com/lilab-stanford/MUSK
# Copyright (c) 2025 Stanford University, by Jinxi Xiang
# Licensed under the CC-BY-NC-ND 4.0 License (https://creativecommons.org/licenses/by-nc-nd/4.0/)
# Please see LICENSE for additional details.
# --------------------------------------------------------

import torch
import huggingface_hub
import os
from safetensors.torch import load_file
import math
import torch.nn.functional as F
from einops import rearrange


def xlm_tokenizer(tokens, tokenizer, max_len=100):
    tokens = tokenizer.encode(tokens)

    tokens = tokens[1:-1]  # remove eos and bos;
    if len(tokens) > max_len - 2:
        tokens = tokens[:max_len - 2]

    tokens = [tokenizer.bos_token_id] + tokens[:] + [tokenizer.eos_token_id]
    num_tokens = len(tokens)
    padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)

    text_tokens = tokens + [tokenizer.pad_token_id] * (max_len - num_tokens)
    return text_tokens, padding_mask


def split_chessboard(x, num_split):
    """
        x: b * c * h * w
        Deividing x into num_split**2 sub-squares, and concatenate all the sub-squares on the batch dimension
    """
    B, C, H, W = x.shape
    assert H % num_split == 0 and W % num_split == 0
    x_split = rearrange(x, 'b c (nh h) (nw w) -> (nh nw b) c h w', nh=num_split, nw=num_split)
    return x_split


def batched_forward(model, x, batch_size=-1):
    x_batched = x.split(batch_size)
    outs = []
    for x in x_batched:
        ret = model(
        image=x,
        out_norm=False,
        with_head=False,
        return_global=True,
        ms_aug=False
        )[0]
        outs.append(ret)
    return torch.cat(outs, dim=0)


"""
During MUSK pretraining, we used multi-scale image inputs, i.e., 
incorporating mixed magnifications. 
To align with this, we leverage MUSK's multi-scale capability during inference 
for linear probe and MIL tasks. Multi-scale was not applied to zero-shot tasks, 
as it is the CLS token that was used for modality alignment in contrastive learning.

The code implementation of MultiScaleForward() is derived from: ArXiv 2024 Vol. abs/2403.13043 
"""
def MultiScaleForward(
        model, 
        input, 
        scales=[1,2], 
        max_split_size=None, 
        ):
    
    assert input.dim() == 4, "Input image must be in the shape of BxCxHxW."
    assert input.shape[2] == input.shape[3], "Currently only square images are supported."

    b, c, input_size, _ = input.shape
    
    # image size for each scale
    img_sizes = [int(input_size * scale) for scale in scales]

    # prepare multiscale inputs
    max_split_size = max_split_size or input_size   # The maximum size of each split of image. Set as the input size by default
    num_splits = [math.ceil(size / max_split_size) for size in img_sizes]   # number of splits each scale
    input_multiscale = []
    for size, num_split in zip(img_sizes, num_splits):
        x = F.interpolate(input.to(torch.float32), size=size, mode='bicubic').to(input.dtype)
        x = split_chessboard(x, num_split=num_split)
        input_multiscale.append(x)

    # run feedforward on each scale
    outs_multiscale = [batched_forward(model, x, b) for x in input_multiscale]
    
    up_scale = rearrange(outs_multiscale[1], '(n b) c -> b n c', b=outs_multiscale[0].shape[0])
    out = torch.cat([outs_multiscale[0], up_scale.mean(1)], dim=-1)
    return out
    

def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))



# The implementation code is modified from DeiT (https://github.com/facebookresearch/deit.git)
def load_model_and_may_interpolate(
        ckpt_path, 
        model,
        model_key, 
        model_prefix, 
        local_dir: str = os.path.join(os.path.expanduser("~"), ".cache/")
        ):
    
    if ckpt_path.startswith("hf_hub:"):
        local_path = os.path.join(local_dir, "model.safetensors")
        if not os.path.exists(local_path):    
            hub_name = ckpt_path.split(":")[1]
            huggingface_hub.hf_hub_download(
                hub_name, 
                filename="model.safetensors", 
                local_dir=local_dir, 
                force_download=True
                )
        
    else:
        local_path = ckpt_path
    
    checkpoint = load_file(local_path)

    print("Load ckpt from %s" % ckpt_path)
    checkpoint_model = None
    for model_key in model_key.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break

    if checkpoint_model is None:
        checkpoint_model = checkpoint

    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    for pos_embed_key in ("vision_pos_embed", "pos_embed", "beit3.encoder.embed_positions.A.weight"):
        if pos_embed_key in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model[pos_embed_key]
            embedding_size = pos_embed_checkpoint.shape[-1]
            if pos_embed_key == "beit3.encoder.embed_positions.A.weight":
                # being consistent with Fairseq, which starts from 2 for position embedding
                torchscale_model = True
                num_patches = model.beit3.vision_embed.num_patches
                num_extra_tokens = model.beit3.vision_embed.num_position_embeddings() + 2 - num_patches
            else:
                torchscale_model = False
                num_patches = model.patch_embed.num_patches
                num_extra_tokens = getattr(model, pos_embed_key).shape[-2] - num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches ** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                if torchscale_model:
                    extra_tokens = pos_embed_checkpoint[:num_extra_tokens].unsqueeze(0)
                    # only the position tokens are interpolated
                    pos_tokens = pos_embed_checkpoint[num_extra_tokens:]
                else:
                    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                    # only the position tokens are interpolated
                    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)

                # interpolate must be carried out on float
                pos_token_type = pos_tokens.dtype
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens.float(), size=(new_size, new_size), mode='bicubic', align_corners=False).to(
                    dtype=pos_token_type)

                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                if torchscale_model:
                    new_pos_embed = new_pos_embed.squeeze(0)
                checkpoint_model[pos_embed_key] = new_pos_embed

    load_state_dict(model, checkpoint_model, prefix=model_prefix)

