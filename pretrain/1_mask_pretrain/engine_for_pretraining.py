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
import math
import sys
from typing import Iterable

import torch
import torch.nn as nn
import os
import utils
from accelerate import Accelerator
import time
import torch.distributed as dist
import datetime
import numpy as np


def train_one_epoch(accelerator: Accelerator, model: torch.nn.Module, d_vae: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, gradient_accumulation_steps=1, param_list=None):
    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10 * gradient_accumulation_steps

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration

        if it % gradient_accumulation_steps == 0:
            if lr_schedule_values is not None or wd_schedule_values is not None:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it // gradient_accumulation_steps] * param_group["lr_scale"]
                    if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule_values[it // gradient_accumulation_steps]

        # get token_id for images in mim
        with torch.inference_mode():
            images = batch["second_image"].to(device, non_blocking=True, dtype=torch.bfloat16)
            input_ids = d_vae.get_codebook_indices(images).flatten(1)
            image_masks = batch["image_mask"].to(device, non_blocking=True, dtype=torch.bfloat16)
            image_masks = image_masks.flatten(1).to(torch.bool)
            image_labels = input_ids[image_masks]
            batch.update({"image_labels": image_labels})

        bsz = batch["image"].shape[0]

        ret = model(batch, device)

        loss_dict = {k: v for k, v in ret.items() if "loss" in k}
        acc_dict = {k: v for k, v in ret.items() if "accuracy" in k}
        loss = sum([v for k, v in loss_dict.items()])

        loss_value = accelerator.gather(loss.repeat(bsz)).mean().item()  # gather loss for log
        loss_dict.update({"loss": loss_value})

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # use accelerator
        accelerator.backward(loss, retain_graph=True)
        accelerator.clip_grad_norm_(param_list, 1.0)
        if it % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        metric_logger.update(**acc_dict)
        metric_logger.update(**loss_dict)

        # Log to wandb by calling `accelerator.log`, `step` is optional
        accelerator.log(acc_dict, )
        accelerator.log(loss_dict, )

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        # metric_logger.update(grad_norm=grad_norm)

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.inference_mode()
def eval_retrieval(model, text_tokenizer, data_loader, device, config, args):
    model.eval()
    print('Computing features for evaluation...')
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_embeds = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]

        text_input = text_tokenizer(text,
                                    padding='max_length',
                                    truncation=True,
                                    max_length=config["max_text_len"],
                                    return_tensors="pt").to(device)

        batch = {"it_text_ids": text_input.input_ids,
                 "it_text_labels": torch.full_like(text_input.input_ids, -100),
                 "it_text_masks": text_input.attention_mask}

        with torch.cuda.amp.autocast():
            # infer text embeddings
            text_output = model.extract_text_features(batch, mask_text=False, device=device)
        text_embed = text_output["cls_feats"]
        text_embeds.append(text_embed)

    image_feats = []
    for image, img_id in data_loader:
        batch = {"it_image": image}

        with torch.cuda.amp.autocast():
            image_output = model.infer_image_ft(batch, mask_image=False, device=device)

        image_feat = image_output["cls_feats"]
        image_feats.append(image_feat)

    image_feats = torch.cat(image_feats, dim=0)
    text_embeds = torch.cat(text_embeds, dim=0)

    all_image_features = image_feats
    all_text_features = text_embeds

    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(all_image_features, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(all_text_features, op=torch.distributed.ReduceOp.SUM)

    sims_matrix = all_image_features @ all_text_features.t()  # image-to-text similarity

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    return sims_matrix.cpu().numpy(), sims_matrix.t().cpu().numpy()


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {'txt_r1': tr1,
                   'txt_r5': tr5,
                   'txt_r10': tr10,
                   'txt_r_mean': tr_mean,
                   'img_r1': ir1,
                   'img_r5': ir5,
                   'img_r10': ir10,
                   'img_r_mean': ir_mean,
                   'r_mean': r_mean}
    return eval_result
