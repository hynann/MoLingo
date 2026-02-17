import sys
sys.path.append('.')

import os
import argparse
import torch
import math
import copy
import numpy as np

from os.path import join as pjoin
from torch.utils.data import DataLoader
from argparse import Namespace
from collections import OrderedDict

import mogen.models.molingo.molingo as molingo_models

from mogen.utils.fixseed import fixseed
from mogen.utils.get_opt import get_opt
from mogen.utils.eval_utils import load_ms_evaluators
from mogen.models.eval.tmr_eval_wrapper import TMREvaluatorModelWrapper
from mogen.models.eval.mardm_evaluators import MARDMEvaluators
from mogen.utils.word_vectorizer import WordVectorizer

from mogen.data.t2m_dataset import Text2MotionDatasetEval, collate_fn
from mogen.data.ms_dataset import Text2MotionDatasetMS
from mogen.models.vae.vae import VAE
from mogen.core.eval import eval_molingo, eval_molingo_ms

def load_vae_model(vae_opt, ckpt_path, dim_pose, device):
    vae_model = VAE(
        input_width=dim_pose,
        output_emb_width=vae_opt.output_emb_width,
        down_t=vae_opt.down_t,
        stride_t=vae_opt.stride_t,
        width=vae_opt.width,
        depth=vae_opt.depth,
        dilation_growth_rate=vae_opt.dilation_growth_rate,
        activation=vae_opt.activation,
        norm=vae_opt.norm,
        pad_mode=vae_opt.pad_mode,
        ae=vae_opt.ae,
    )
    vae_model.to(device)
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_key = k[len("module."):]
        else:
            new_key = k
        new_state_dict[new_key] = v

    vae_model.load_state_dict(state_dict=new_state_dict)
    for param in vae_model.parameters():
        param.requires_grad = False
    return vae_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configurations for the Model Eval")
    parser.add_argument("-d", "--dim_pose", type=int, default=263, choices=[263, 272], help="Motion representation")
    parser.add_argument("-dr", "--data_root", type=str, default='/home/hynann/data', help="Modify with your own dataset directory")
    parser.add_argument("-s", "--step", type=int, default=32, help="The number of Rectified Flow sampling steps")
    parser.add_argument("-c", "--cfg", type=float, default=5.5, help="CFG value")
    parser.add_argument("-t", "--temperature", type=float, default=1.0, help="CFG temprature")
    parser.add_argument("-a", "--acc", type=int, default=3,
                        help="Sampling reduction factor. Number of steps = total_tokens // acc (e.g., 49 tokens with acc=3 → 16 steps).")
    parser.add_argument("-r", "--repeat", type=int, default=1)
    parser.add_argument('-cm', "--cal_mm", action='store_true', help="Whether calculate Multi-Modality metric or not")

    args = parser.parse_args()

    fixseed(3407)

    device = 'cuda'
    data_root = args.data_root
    dim_pose = args.dim_pose
    step = args.step
    cfg = args.cfg
    acc = args.acc
    repeat_times = args.repeat

    opt = Namespace()
    opt.dataset_name = 't2m' if dim_pose == 263 else 'ms'

    # create eval output file
    model_dir = pjoin('mogen/checkpoints', opt.dataset_name, f'pretrained_model_{dim_pose}')
    opt_path = pjoin(model_dir, 'opt.txt')
    model_opt = get_opt(opt_path, device)

    eval_output_dir = pjoin(model_dir, f'eval_cfg_{cfg}_step_{step}_acc_{acc}')
    os.makedirs(eval_output_dir, exist_ok=True)
    eval_file = pjoin(eval_output_dir, 'eval_res.txt')

    # data path configuration
    opt.joints_num = 22

    if not dim_pose == 272:
        opt.data_root = pjoin(data_root, 'HumanML3D/HumanML3D')
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        fps = 20
    else:
        opt.data_root = pjoin(data_root, 'HumanML3D_272')
        opt.motion_dir = pjoin(opt.data_root, 'motion_data')
        fps = 30
    opt.text_dir = pjoin(opt.data_root, 'texts')

    # eval wrapper initialization
    if not dim_pose == 272: # load evaluators for MARDM-67 and TMR-263
        tmr_wrapper = TMREvaluatorModelWrapper(device=device)
        mardm_wrapper = MARDMEvaluators('t2m', device=device)
    else: # load evaluator for MS-272
        textencoder, motionencoder = load_ms_evaluators(device=device)

    # load vae
    vae_opt_path = pjoin('mogen/checkpoints', opt.dataset_name, model_opt.vae, 'opt.txt')
    vae_ckpt_path = pjoin('mogen/checkpoints', opt.dataset_name, model_opt.vae, 'model', 'net_best_fid.ckpt')
    vae_opt = get_opt(vae_opt_path, device=device)
    vae_model = load_vae_model(vae_opt, vae_ckpt_path, dim_pose, device=device)

    vae_embed_dim = vae_opt.output_emb_width
    ds_rate = math.pow(2, vae_opt.down_t)
    ds_rate = int(ds_rate)
    opt.unit_length = ds_rate

    # test dataset
    if not dim_pose == 272:
        mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
        std = np.load(pjoin(opt.data_root, 'Std.npy'))
        mean_eval_mardm = np.load('./mogen/utils/eval_mean_std_mardm/eval_mean.npy')
        std_eval_mardm = np.load('./mogen/utils/eval_mean_std_mardm/eval_std.npy')
        test_split_file = pjoin(opt.data_root, 'test.txt')
        w_vectorizer = WordVectorizer('./glove', 'our_vab')
        opt.max_motion_length = 196
        opt.max_text_len = 20
        test_dataset = Text2MotionDatasetEval(opt, mean, std, mean_eval_mardm, std_eval_mardm, test_split_file, w_vectorizer)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=4, collate_fn=collate_fn)
    else:
        mean = np.load(pjoin(opt.data_root, 'mean_std', 'Mean.npy'))
        std = np.load(pjoin(opt.data_root, 'mean_std', 'Std.npy'))
        test_split_file = pjoin(opt.data_root, 'split', 'test.txt')
        opt.max_motion_length = 300
        test_dataset = Text2MotionDatasetMS(opt, mean, std, test_split_file)
        test_loader = DataLoader(test_dataset, batch_size=32, drop_last=True, shuffle=True, num_workers=4, pin_memory=True)

    # molingo model initialization
    model_func_name = f'molingo_{model_opt.model_size}'
    molingo_func = getattr(molingo_models, model_func_name)
    partial_molingo = molingo_func()

    molingo_model = partial_molingo(vae_embed_dim=vae_embed_dim,
                              token_size=opt.max_motion_length // ds_rate,
                              unit_length=ds_rate,
                              sample_steps=step,
                              t5_max_len=model_opt.t5_max_len,
                              adapter_layers=model_opt.aligner_layers,
                              ae=vae_opt.ae,
                              )
    molingo_model.to(device)
    model_without_ddp = molingo_model

    # load molingo model
    checkpoint = torch.load(pjoin(model_dir, f"net_best_fid.pth"), map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])
    model_params = list(model_without_ddp.parameters())
    ema_state_dict = checkpoint['model_ema']
    ema_params = [ema_state_dict[name].cuda() for name, _ in model_without_ddp.named_parameters()]
    del checkpoint
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = ema_params[i]
    print("Switch to ema")
    model_without_ddp.load_state_dict(ema_state_dict)
    del ema_state_dict
    model_without_ddp.eval()

    # eval
    if not dim_pose == 272:
        fid_tmr, top1_tmr, top2_tmr, top3_tmr, matching_score_tmr, = [], [], [], [], []
        fid_mardm, top1_mardm, top2_mardm, top3_mardm, matching_score_mardm = [], [], [], [], []
        clip_score_mardm = []
        mm_tmr, mm_mardm = [], []

        with open(eval_file, 'w') as f:
            for rt in range(repeat_times):
                tmp_fid_tmr, tmp_top1_tmr, tmp_top2_tmr, tmp_top3_tmr, \
                tmp_matching_score_tmr,tmp_fid_mardm, tmp_top1_mardm, \
                tmp_top2_mardm, tmp_top3_mardm, tmp_matching_score_mardm, \
                tmp_clip_score, \
                multimodality_tmr, multimodality_mardm = eval_molingo(test_loader, model_without_ddp, vae_model,
                                            test_dataset, ep=rt, cfg=cfg,
                                            tmr_wrapper=tmr_wrapper, mardm_wrapper=mardm_wrapper,
                                            std_factor=model_opt.std_factor, acc_ratio=acc, cal_mm=args.cal_mm)
                fid_tmr.append(tmp_fid_tmr)
                top1_tmr.append(tmp_top1_tmr)
                top2_tmr.append(tmp_top2_tmr)
                top3_tmr.append(tmp_top3_tmr)
                matching_score_tmr.append(tmp_matching_score_tmr)
                fid_mardm.append(tmp_fid_mardm)
                top1_mardm.append(tmp_top1_mardm)
                top2_mardm.append(tmp_top2_mardm)
                top3_mardm.append(tmp_top3_mardm)
                matching_score_mardm.append(tmp_matching_score_mardm)
                clip_score_mardm.append(tmp_clip_score)
                mm_tmr.append(multimodality_tmr)
                mm_mardm.append(multimodality_mardm)

            fid_tmr = np.array(fid_tmr)
            top1_tmr = np.array(top1_tmr)
            top2_tmr = np.array(top2_tmr)
            top3_tmr = np.array(top3_tmr)
            matching_score_tmr = np.array(matching_score_tmr)
            fid_mardm = np.array(fid_mardm)
            top1_mardm = np.array(top1_mardm)
            top2_mardm = np.array(top2_mardm)
            top3_mardm = np.array(top3_mardm)
            matching_score_mardm = np.array(matching_score_mardm)
            clip_score_mardm = np.array(clip_score_mardm)
            multimodality_tmr = np.array(multimodality_tmr)
            multimodality_mardm = np.array(multimodality_mardm)

            msg_final_tmr = f"\tTMR FID: {np.mean(fid_tmr):.3f}, conf. {np.std(fid_tmr) * 1.96 / np.sqrt(repeat_times):.3f}\n" \
                            f"\tTOP1: {np.mean(top1_tmr):.3f}, conf. {np.std(top1_tmr) * 1.96 / np.sqrt(repeat_times):.3f}, TOP2. {np.mean(top2_tmr):.3f}, conf. {np.std(top2_tmr) * 1.96 / np.sqrt(repeat_times):.3f}, TOP3. {np.mean(top3_tmr):.3f}, conf. {np.std(top3_tmr) * 1.96 / np.sqrt(repeat_times):.3f}\n" \
                            f"\tMatching Score: {np.mean(matching_score_tmr):.3f}, conf. {np.std(matching_score_tmr) * 1.96 / np.sqrt(repeat_times):.3f}\n" \
                            f"\tMultimodality: {np.mean(multimodality_tmr): .3f}, conf. {np.std(multimodality_tmr) * 1.96 / np.sqrt(repeat_times):.3f}\n"
            print(msg_final_tmr)
            print(msg_final_tmr, file=f, flush=True)

            msg_final_mardm = f"\tMARDM FID: {np.mean(fid_mardm):.3f}, conf. {np.std(fid_mardm) * 1.96 / np.sqrt(repeat_times):.3f}\n" \
                            f"\tTOP1: {np.mean(top1_mardm):.3f}, conf. {np.std(top1_mardm) * 1.96 / np.sqrt(repeat_times):.3f}, TOP2. {np.mean(top2_mardm):.3f}, conf. {np.std(top2_mardm) * 1.96 / np.sqrt(repeat_times):.3f}, TOP3. {np.mean(top3_mardm):.3f}, conf. {np.std(top3_mardm) * 1.96 / np.sqrt(repeat_times):.3f}\n" \
                            f"\tMatching Score: {np.mean(matching_score_mardm):.3f}, conf. {np.std(matching_score_mardm) * 1.96 / np.sqrt(repeat_times):.3f}\n" \
                            f"\tClip Score: {np.mean(clip_score_mardm):.3f}, conf. {np.std(clip_score_mardm) * 1.96 / np.sqrt(repeat_times):.3f}\n" \
                            f"\tMultimodality: {np.mean(multimodality_mardm): .3f}, conf. {np.std(multimodality_mardm) * 1.96 / np.sqrt(repeat_times):.3f}\n"

            print(msg_final_mardm)
            print(msg_final_mardm, file=f, flush=True)
    else:
        fid_ms, top1_ms, top2_ms, top3_ms, matching_score_ms, = [], [], [], [], []
        model_without_ddp.eval()

        with open(eval_file, 'w') as f:
            for rt in range(repeat_times):
                tmp_fid_ms, tmp_top1_ms, tmp_top2_ms, tmp_top3_ms, \
                    tmp_matching_score_ms, = eval_molingo_ms(test_loader, model_without_ddp, vae_model,
                                                           ep=rt, cfg=cfg,
                                                           motionencoder=motionencoder, textencoder=textencoder,
                                                           std_factor=model_opt.std_factor, acc_ratio=acc)

                fid_ms.append(tmp_fid_ms)
                top1_ms.append(tmp_top1_ms)
                top2_ms.append(tmp_top2_ms)
                top3_ms.append(tmp_top3_ms)
                matching_score_ms.append(tmp_matching_score_ms)

            fid_ms = np.array(fid_ms)
            top1_ms = np.array(top1_ms)
            top2_ms = np.array(top2_ms)
            top3_ms = np.array(top3_ms)
            matching_score_ms = np.array(matching_score_ms)

            msg_final_ms = f"\tTMR FID: {np.mean(fid_ms):.3f}, conf. {np.std(fid_ms) * 1.96 / np.sqrt(repeat_times):.3f}\n" \
                            f"\tTOP1: {np.mean(top1_ms):.3f}, conf. {np.std(top1_ms) * 1.96 / np.sqrt(repeat_times):.3f}, TOP2. {np.mean(top2_ms):.3f}, conf. {np.std(top2_ms) * 1.96 / np.sqrt(repeat_times):.3f}, TOP3. {np.mean(top3_ms):.3f}, conf. {np.std(top3_ms) * 1.96 / np.sqrt(repeat_times):.3f}\n" \
                            f"\tMatching Score: {np.mean(matching_score_ms):.3f}, conf. {np.std(matching_score_ms) * 1.96 / np.sqrt(repeat_times):.3f}\n"
            print(msg_final_ms)
            print(msg_final_ms, file=f, flush=True)

    print('Done calculating metrics')










