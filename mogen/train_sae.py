import sys
sys.path.append('.')
import os
from os.path import join as pjoin
import math
import numpy as np
from torch.utils.data import DataLoader

from mogen.utils import paramUtil
from mogen.options.sae_option import arg_parse
from mogen.utils.fixseed import fixseed
from mogen.utils.eval_utils import load_ms_evaluators
from mogen.data.ms_dataset import Text2MotionDatasetMSBabel
from mogen.models.vae.vae import VAE
from mogen.core.sae_trainer import SAETrainer

if __name__ == '__main__':
    opt = arg_parse(True)
    print(f"Using Device: {opt.device}")

    fixseed(opt.seed)

    opt.save_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_dir, 'model')
    opt.ani_dir = pjoin(opt.save_dir, 'animation')

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.ani_dir, exist_ok=True)

    opt.data_dir = pjoin(opt.data_root, 'HumanML3D_272')
    opt.motion_dir = pjoin(opt.data_dir, 'motion_data')
    opt.text_dir = pjoin(opt.data_dir, 'texts')
    opt.babel_dir = pjoin(opt.data_dir, 'babel_272_annotation_t5')
    dim_pose = 272
    fps = 30
    kinematic_chain = paramUtil.t2m_kinematic_chain
    radius = 4

    ds_rate = math.pow(2, opt.down_t)
    ds_rate = int(ds_rate)
    opt.unit_length = ds_rate
    opt.max_motion_length = 300

    # eval wrapper initialization
    _, motionencoder = load_ms_evaluators(device=opt.device)

    mean = np.load(pjoin(opt.data_dir, 'mean_std', 'Mean.npy'))
    std = np.load(pjoin(opt.data_dir, 'mean_std', 'Std.npy'))

    train_split_file = pjoin(opt.data_dir, 'split', 'train.txt')
    val_split_file = pjoin(opt.data_dir, 'split', 'val.txt')

    train_dataset = Text2MotionDatasetMSBabel(opt, mean, std, train_split_file)
    val_dataset = Text2MotionDatasetMSBabel(opt, mean, std, val_split_file)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=8,
                              shuffle=True, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=8,
                             shuffle=True, pin_memory=False)

    # build network
    sae = VAE(
        input_width=dim_pose,
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

    # build trainer
    trainer = SAETrainer(args=opt,
                        vae_model=sae,
                        datamodule=train_dataset,
                        ms_wrapper=motionencoder)

    trainer.train(train_loader, val_loader)