import sys
sys.path.append('.')
import math
import os
from os.path import join as pjoin

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import mogen.models.molingo.molingo as molingo_models
import mogen.utils.misc as misc
from mogen.core.molingo_trainer import MoLingoTrainer
from mogen.data.ms_dataset import Text2MotionDatasetMS
from mogen.models.vae.vae import VAE
from mogen.options.molingo_option import arg_parse
from mogen.utils import paramUtil
from mogen.utils.eval_utils import load_ms_evaluators
from mogen.utils.fixseed import fixseed
from mogen.utils.get_opt import get_opt
from mogen.utils.plot_script import plot_t2m

os.environ["OMP_NUM_THREADS"] = "1"


def load_vae_model(opt, ckpt_path, device):
    vae_model = VAE(
        input_width=272,
        output_emb_width=opt.output_emb_width,
        down_t=opt.down_t,
        stride_t=opt.stride_t,
        width=opt.width,
        depth=opt.depth,
        dilation_growth_rate=opt.dilation_growth_rate,
        activation=opt.activation,
        norm=opt.norm,
        pad_mode=opt.pad_mode,
        ae=opt.ae,
    )
    vae_model.to(device)
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    vae_model.load_state_dict(state_dict=state_dict)
    return vae_model


if __name__ == "__main__":
    opt = arg_parse(is_train=True)

    misc.init_distributed_mode(opt)
    print('finished setting the distributed mode')

    opt.device = "cuda"

    fixseed(opt.seed)
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    dim_pose = 272

    print(f'num tasks: {num_tasks}; global rank: {global_rank}')

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin(opt.save_root, 'log')

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    opt.data_dir = pjoin(opt.data_root, 'HumanML3D_272')
    opt.motion_dir = pjoin(opt.data_dir, 'motion_data')
    opt.text_dir = pjoin(opt.data_dir, 'texts')
    opt.joints_num = 22
    fps = 30
    radius = 4
    kinematic_chain = paramUtil.t2m_kinematic_chain

    # load vae
    vae_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vae, 'opt.txt')
    ckpt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vae, 'model', 'net_best_fid.ckpt')
    vae_opt = get_opt(vae_opt_path, device=opt.device)

    vae_model = load_vae_model(vae_opt, ckpt_path, device=opt.device)
    for param in vae_model.parameters():
        param.requires_grad = False

    textencoder, motionencoder = load_ms_evaluators(device=opt.device)

    vae_embed_dim = vae_opt.output_emb_width
    ds_rate = int(math.pow(2, vae_opt.down_t))
    opt.unit_length = ds_rate
    opt.max_motion_length = 300

    # dataset
    mean = np.load(pjoin(opt.data_dir, 'mean_std', 'Mean.npy'))
    std = np.load(pjoin(opt.data_dir, 'mean_std', 'Std.npy'))

    train_split_file = pjoin(opt.data_dir, 'split', 'train.txt')
    val_split_file = pjoin(opt.data_dir, 'split', 'val.txt')

    train_dataset = Text2MotionDatasetMS(opt, mean, std, train_split_file)
    eval_val_dataset = Text2MotionDatasetMS(opt, mean, std, val_split_file)

    train_sampler = DistributedSampler(train_dataset, num_replicas=num_tasks, rank=global_rank)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, sampler=train_sampler, drop_last=True,
                              num_workers=4, pin_memory=True)
    eval_val_loader = DataLoader(eval_val_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                 num_workers=4, pin_memory=True)

    # initialize molingo
    partial_molingo = getattr(molingo_models, f'molingo_{opt.model_size}')()
    molingo_model = partial_molingo(
        vae_embed_dim=vae_embed_dim,
        token_size=opt.max_motion_length // ds_rate,
        unit_length=ds_rate,
        t5_max_len=opt.t5_max_len,
        adapter_layers=opt.adapter_layers,
        label_drop_prob=opt.label_drop_prob,
        proj_dropout=opt.proj_dropout,
        flow_batch_mul=opt.flow_batch_mul,
        sample_steps=opt.sample_steps,
        grad_checkpointing=opt.grad_checkpointing,
        ae=vae_opt.ae,
    )
    molingo_model.to(opt.device)

    model_without_ddp = molingo_model

    if opt.distributed:
        molingo_model = DDP(molingo_model, device_ids=[opt.gpu])
        model_without_ddp = molingo_model.module

    # base_lr scaling
    eff_batch_size = opt.batch_size * misc.get_world_size()
    opt.base_lr = opt.base_lr * eff_batch_size / 256

    trainer = MoLingoTrainer(opt, molingo_model, model_without_ddp, vae_model, eval_val_dataset,
                             rank=global_rank, ae=vae_opt.ae)
    trainer.train(train_loader, eval_val_loader, motionencoder, textencoder, plot_eval=plot_t2m)
    dist.destroy_process_group()
