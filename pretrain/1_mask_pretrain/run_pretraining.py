import argparse
import datetime
import numpy as np
import time
import torch
import json
import os
from utils import setup_for_distributed, get_sentencepiece_model
from pathlib import Path
import random
import types
from timm.models import create_model
from optim_factory import create_optimizer
from datasets.musk_datasets import build_dataset, build_quilt1m_dataset
from engine_for_pretraining import train_one_epoch
import utils
from ruamel.yaml import YAML
from accelerate import Accelerator
from argparse import Namespace

import modeling_musk
import modeling_vqkd

def get_args():
    parser = argparse.ArgumentParser('MSUK pre-training script', add_help=False)
    parser.add_argument('--config', default='', type=str)
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    return parser.parse_args()


def get_model(model_name, drop_path, mask_token):
    print(f"Creating model: {model_name}")
    model = create_model(
        model_name,
        pretrained=False,
        drop_path_rate=drop_path,
        vocab_size=64010,
        v_vocab_size=8192,
        mask_token=mask_token,
        checkpoint_activations=True,
    )
    return model

def get_visual_tokenizer(args):
    print(f"Creating visual tokenizer: {args.tokenizer_model}")
    model = create_model(
            args.tokenizer_model,
            img_size=args.second_input_size,
            pretrained=True,
            pretrained_weight=args.tokenizer_weight,
            as_tokenzer=True,
            n_code=args.codebook_size, 
            code_dim=args.codebook_dim,
        ).eval()
    return model


def main(config):

    accelerator = Accelerator(
        gradient_accumulation_steps=config['general']['gradient_accumulation_steps'],
        # log_with="wandb",
    )
    # accelerator.init_trackers(
    #     project_name=f"musk_image_tokenizer",
    #     config=config,
    #     init_kwargs={"wandb": {"name": config['log_dir'].split("/")[-2]}},
    # )
    
    setup_for_distributed(accelerator.is_main_process)
    print(accelerator.state)
    device = accelerator.device
    
    if accelerator.state.deepspeed_plugin is not None and accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] == "auto":
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = config['dataset']['batch_size']

    # fix the seed for reproducibility
    seed = config['general']['seed'] + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # create text tokenizer
    tokenizer = get_sentencepiece_model(config['model']['tokenizer'])
    
    mask_token = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    model = get_model(config['model']['name'], config['model']['drop_path'], mask_token)
    if config['general']['resume']:
        utils.load_model_and_may_interpolate(config['general']['resume'], model, 'model|module','')

    patch_size = config['model']['patch_size']
    print("Patch size = %s" % str(patch_size))
    config['model']['window_size'] = (config['model']['input_size'] // patch_size, config['model']['input_size'] // patch_size)

    if accelerator.state.deepspeed_plugin is not None:
        config['general']['gradient_accumulation_steps'] = accelerator.state.deepspeed_plugin.deepspeed_config["gradient_accumulation_steps"]
        print(f"gradient_accumulation_steps: {config['general']['gradient_accumulation_steps']}")

    
    image_dir = config['dataset']['image_dir']
    text_dir = config['dataset']['text_dir']


    dataset_train = build_dataset(
        input_size=config['model']['input_size'],
        second_input_size=config['visual_tokenizer']['second_input_size'],
        window_size=config['model']['window_size'],
        max_text_len=config['model']['max_text_len'],
        image_dir=image_dir,
        text_dir=text_dir,
        tokenizer=tokenizer,
        image_index=[1, 0],
        text_index=[1, 0],
        num_mask_patches=config['model']['num_mask_patches'],
        min_mask_patches_per_block=config['model']['min_mask_patches_per_block'],
        max_mask_patches_per_block=config['model']['max_mask_patches_per_block']
        )
    
    # prepare discrete vae
    v_tokenizer_args = types.SimpleNamespace(**config['visual_tokenizer'])
    d_vae = get_visual_tokenizer(v_tokenizer_args)
    d_vae.to(device, dtype=torch.bfloat16)

    num_tasks = accelerator.num_processes
    sampler_rank = accelerator.process_index
    num_training_steps_per_epoch = len(dataset_train) // config['dataset']['batch_size'] // num_tasks

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=False
    )
    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config['dataset']['batch_size'],
        num_workers=config['dataset']['num_workers'],
        pin_memory=config['dataset']['pin_mem'],
        drop_last=True,
    )

    model.to(device, dtype=torch.bfloat16)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print('number of params:{:.3f} M'.format(n_parameters / 1e6))
    
    total_batch_size = config['dataset']['batch_size'] * accelerator.num_processes
    print("LR = %.8f" % config['optimizer']['lr'])
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    model_without_ddp = model
    opt_args = types.SimpleNamespace(**config['optimizer'])
    optimizer = create_optimizer(opt_args, model_without_ddp)
    loss_scaler = None  # for deepspeed checkpoint saving
    # loss_scaler = NativeScaler()  # for ddp checkpoint saving

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        config['optimizer']['lr'], config['optimizer']['min_lr'], config['general']['epochs'], num_training_steps_per_epoch,
        warmup_epochs=config['optimizer']['warmup_epochs'], warmup_steps=config['optimizer']['warmup_steps'],
    )

    # warp with accelerator
    model, optimizer = accelerator.prepare(model, optimizer)
    load_args = Namespace(**{'resume': config['general']['resume'], 
                             'auto_resume': config['general']['auto_resume'], 
                             'output_dir': config['output_dir'],
                             'start_epoch': config['general']['start_epoch']
                             })
    
    utils.auto_load_model(args=load_args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    config['general']['start_epoch'] = load_args.start_epoch

    wd_schedule_values = utils.cosine_scheduler(
        config['optimizer']['weight_decay'], 
        config['optimizer']['weight_decay_end'], 
        config['general']['epochs'], 
        num_training_steps_per_epoch
        )
    
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    print(f"Start training for {config['general']['epochs']} epochs")
    start_time = time.time()

    for epoch in range(config['general']['start_epoch'], config['general']['epochs']):

        train_stats = train_one_epoch(
            accelerator=accelerator,
            model=model,
            d_vae=d_vae, data_loader=data_loader_train,
            optimizer=optimizer, device=device, epoch=epoch,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            gradient_accumulation_steps=config['general']['gradient_accumulation_steps'],
            param_list=model_without_ddp.parameters()
        )

        # --------------- save checkpoint and training logs --------------- #
        if ((epoch + 1) % config['general']['save_ckpt_freq'] == 0 or epoch + 1 == config['general']['epochs']):
            # deepspeed save
            utils.save_model(args=config, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=None, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch, 'n_parameters': n_parameters}

        if config['output_dir']  and utils.is_main_process():
            with open(os.path.join(config['output_dir'] , "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':

    opts = get_args()

    with open(opts.config, "r") as f:
        yaml = YAML(typ='safe', pure=True)
        config = yaml.load(f)
    
    config['log_dir'] = opts.log_dir
    config['output_dir'] = opts.output_dir

    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)

    main(config)
