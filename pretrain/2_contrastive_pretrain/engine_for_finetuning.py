# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import math
import sys
import json
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.utils import accuracy, ModelEma
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from dataset.datasets import get_sentencepiece_model_for_beit3
import numpy as np
import utils
from PIL import Image


class TaskHandler(object):
    def __init__(self) -> None:
        self.metric_logger = None
        self.split = None

    def train_batch(self, model, **kwargs):
        raise NotImplementedError()

    def eval_batch(self, model, **kwargs):
        raise NotImplementedError()

    def before_eval(self, metric_logger, data_loader, **kwargs):
        self.metric_logger = metric_logger
        self.split = data_loader.dataset.split

    def after_eval(self, **kwargs):
        raise NotImplementedError()



class RetrievalHandler(TaskHandler):
    def __init__(self) -> None:
        super().__init__()
        self.image_feats = []
        self.text_feats = []
        self.image_ids = []
        self.metric_logger = None

    def train_batch(self, model, image, language_tokens, padding_mask, image_id):
        loss, vision_cls, language_cls = model(
            image=image, text_description=language_tokens, padding_mask=padding_mask)
        return {
            "loss": loss,
        }

    def before_eval(self, metric_logger, **kwargs):
        self.image_feats.clear()
        self.text_feats.clear()
        self.image_ids.clear()
        self.metric_logger = metric_logger

    def eval_batch(self, model, image, language_tokens, padding_mask, image_id):

        # for beit3 model
        vision_cls, _ = model(image=image, only_infer=True)
        _, language_cls = model(text_description=language_tokens, padding_mask=padding_mask, only_infer=True)

        self.image_feats.append(vision_cls.clone())
        self.text_feats.append(language_cls.clone())
        self.image_ids.append(image_id.clone())

    def after_eval(self, **kwargs):
        
        image_embeds = torch.cat(self.image_feats)
        text_embeds = torch.cat(self.text_feats)

        # >>>>>>>>>>>> image2text >>>>>>>>>>>> #
        text_probs = (100.0 * image_embeds @ text_embeds.T).softmax(dim=-1)

        # torch.save(image_embeds @ text_embeds.T, './outputs/books_set_i2t.pt')

        # Assume sim is your similarity matrix of shape (batch_size, batch_size)
        num_items = image_embeds.shape[0]
        sim_sorted_indices = np.argsort(-text_probs.cpu(), axis=1)  # Sort in descending order
        correct_indices = np.arange(num_items)

        # Assume correct_indices is a 1D array of shape (batch_size,),
        # where correct_indices[i] is the index of the correct text for image i
        correct_ranks = np.array(
            [np.where(row == correct_indices[i])[0][0] for i, row in enumerate(sim_sorted_indices)])

        # np.save('./outputs/correct_ranks_i2t.npy', correct_ranks)

        # Calculate retrieval metrics
        i2t_top1 = np.mean(correct_ranks < 1)
        i2t_top50 = np.mean(correct_ranks < 50)
        i2t_top200 = np.mean(correct_ranks < 200)
        # <<<<<<<<<<<<<< image2text <<<<<<<<<<<<<< #

        # text2image
        image_probs = (100.0 * text_embeds @ image_embeds.T).softmax(dim=-1)
        # torch.save(text_embeds @ image_embeds.T, 'books_set_t2i.pt')

        # Assume sim is your similarity matrix of shape (batch_size, batch_size)
        sim_sorted_indices = np.argsort(-image_probs.cpu(), axis=1)  # Sort in descending order
        correct_indices = np.arange(num_items)

        # Assume correct_indices is a 1D array of shape (batch_size,),
        # where correct_indices[i] is the index of the correct text for image i
        correct_ranks = np.array([np.where(row == correct_indices[i])[0][0]
                                  for i, row in enumerate(sim_sorted_indices)])

        # np.save('correct_ranks_t2i.npy', correct_ranks)

        # Calculate retrieval metrics
        t2i_top1 = np.mean(correct_ranks < 1)
        t2i_top50 = np.mean(correct_ranks < 50)
        t2i_top200 = np.mean(correct_ranks < 200)

        eval_result = {
            "tr_r200": i2t_top200 * 100.0,
            "tr_r50": i2t_top50 * 100.0,
            "tr_r1": i2t_top1 * 100.0,
            "ir_r100": t2i_top200 * 100.0,
            "ir_r50": t2i_top50 * 100.0,
            "ir_r1": t2i_top1 * 100.0,
            "average_score": 100.0 * (i2t_top1 + i2t_top50 + i2t_top200 + t2i_top200 + t2i_top50 + t2i_top1) / 6.0,
        }

        print('* Eval result = %s' % json.dumps(eval_result))
        return eval_result, "average_score"
    

class RetrievalMaskDecoderHandler(RetrievalHandler):

    def train_batch(self, model, image, language_tokens, text_input_ids, padding_mask, image_id):
        
        ct_loss, mlm_loss, mlm_acc = model(
            image=image, text_description=language_tokens, language_labels=text_input_ids, padding_mask=padding_mask,
        )
        mlm_loss = 0.1 * mlm_loss
        return {
            "loss": ct_loss + mlm_loss,
            "ct_loss": ct_loss,
            "mlm_loss": mlm_loss,
            "mlm_acc": mlm_acc,
        }

    def eval_batch(self, model, image, language_tokens, text_input_ids, padding_mask, image_id):
        vision_cls, _ = model(image=image, only_infer=True)
        _, language_cls = model(
            text_description=language_tokens, padding_mask=padding_mask, only_infer=True)

        self.image_feats.append(vision_cls.clone())
        self.text_feats.append(language_cls.clone())
        self.image_ids.append(image_id.clone())


import torch.distributed as dist

def gather_tensor(tensor):
    gather_list = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_list, tensor)
    return torch.cat(gather_list, dim=0)


def get_handler(args):
    
    if args.task in ["quilt1m_retrieval", "quilt1m_pathcap"]:
        return RetrievalMaskDecoderHandler()
    else:
        raise NotImplementedError("Sorry, %s is not support." % args.task)


def train_one_epoch(
        model: torch.nn.Module, data_loader: Iterable,
        optimizer: torch.optim.Optimizer, device: torch.device,
        handler: TaskHandler, epoch: int, start_steps: int,
        lr_schedule_values: list, loss_scaler, max_norm: float = 0,
        update_freq: int = 1, model_ema: Optional[ModelEma] = None,
        log_writer: Optional[utils.TensorboardLogger] = None,
        task=None, mixup_fn=None
):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        global_step = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[global_step] * param_group["lr_scale"]
        # put input data into cuda
        for tensor_key in data.keys():
            data[tensor_key] = data[tensor_key].to(device, non_blocking=True)
            # print("input %s = %s" % (tensor_key, data[tensor_key]))
            if loss_scaler is None and tensor_key.startswith("image"):
                data[tensor_key] = data[tensor_key].half()

            if loss_scaler is None and tensor_key.startswith("moco"):
                data[tensor_key] = data[tensor_key].half()
                
        
        # mixup for imagenet finetuning
        if mixup_fn is not None:
            data["image"], data["label"] = mixup_fn(data["image"], data["label"])

        if task in ["coco_captioning", "nocaps", "quilt1m_captioning"]:
            data["global_step"] = global_step

        if loss_scaler is None:
            results = handler.train_batch(model, **data)
        else:
            with torch.cuda.amp.autocast():
                results = handler.train_batch(model, **data)

        loss = results["loss"]
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = utils.get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(**results)
        metric_logger.update(loss_scale=loss_scale_value)
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
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            kwargs = {
                # "loss": loss_value,
            }
            for key in results:
                kwargs[key] = results[key]
            log_writer.update(head="train", **kwargs)

            kwargs = {
                "loss_scale": loss_scale_value,
                "lr": max_lr,
                "min_lr": min_lr,
                "weight_decay": weight_decay_value,
                "grad_norm": grad_norm,
            }
            log_writer.update(head="opt", **kwargs)
            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, handler):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    handler.before_eval(metric_logger=metric_logger, data_loader=data_loader)

    with torch.inference_mode():

        for data in metric_logger.log_every(data_loader, 10, header):
            for tensor_key in data.keys():
                data[tensor_key] = data[tensor_key].to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                handler.eval_batch(model=model, **data)
            
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    return handler.after_eval()
