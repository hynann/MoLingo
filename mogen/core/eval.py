import sys
sys.path.append('.')

import copy
import os
from os.path import join as pjoin

import torch
import wandb
from tqdm import tqdm as tqdm

import mogen.utils.misc as misc
from mogen.utils.metrics import *

@torch.no_grad()
def eval_molingo(val_loader, model_without_ddp, vae_model, datamodule, ep, cfg,
               tmr_wrapper, mardm_wrapper, temperature=1., std_factor=1., acc_ratio=1., cfg_schedule='constant', cal_mm=True):

    tmr_real_list, tmr_pred_list = [], []
    mardm_real_list, mardm_pred_list = [], []
    motion_multimodality_tmr, motion_multimodality_mardm = [], []

    RP_tmr_real, RP_tmr_pred = 0, 0
    RP_mardm_real, RP_mardm_pred = 0, 0
    multimodality_tmr, multimodality_mardm = 0, 0

    matching_score_tmr_real, matching_score_tmr_pred = 0, 0
    matching_score_mardm_real, matching_score_mardm_pred = 0, 0

    clip_score_real, clip_score_pred = 0, 0

    nb_sample = 0

    if not cal_mm:
        num_mm_batch = 0
    else:
        num_mm_batch = 3

    for i, batch in tqdm(enumerate(val_loader)):
        word_embeddings, pos_one_hots, clip_text, sent_len, feats_ref, m_length, token, hml_keys = batch
        feats_ref = feats_ref.detach().cuda().float()
        m_length = m_length.cuda()
        bs, token_len = feats_ref.shape[:2]
        if i < num_mm_batch:
            motion_multimodality_batch_tmr = []
            motion_multimodality_batch_mardm = []
            for _ in range(30):

                with torch.no_grad():
                    sampled_tokens = model_without_ddp.sample_tokens(bsz=bs, m_lens=m_length, cfg=cfg,
                                                                     cfg_schedule=cfg_schedule, labels=clip_text,
                                                                     temperature=temperature, acc_ratio=acc_ratio)

                    feats_rst = vae_model.decode(sampled_tokens / std_factor)

                feats_rst_denorm = datamodule.inv_transform_torch(feats_rst)
                feats_rst_mardm_eval = datamodule.feats_to_eval_mardm(feats_rst)

                et_tmr_pred, em_tmr_pred = tmr_wrapper.get_co_embeddings(feats_rst_denorm, m_length,
                                                                         clip_text=clip_text)
                (et_mardm_pred, em_mardm_pred), (
                clip_et_mardm_pred, clip_em_mardm_pred) = mardm_wrapper.get_co_embeddings(
                    word_embeddings, pos_one_hots, sent_len, clip_text, feats_rst_mardm_eval, m_length)

                motion_multimodality_batch_tmr.append(em_tmr_pred.unsqueeze(1))
                motion_multimodality_batch_mardm.append(em_mardm_pred.unsqueeze(1))

            motion_multimodality_batch_tmr = torch.cat(motion_multimodality_batch_tmr, dim=1)
            motion_multimodality_batch_mardm = torch.cat(motion_multimodality_batch_mardm, dim=1)
            motion_multimodality_tmr.append(motion_multimodality_batch_tmr)
            motion_multimodality_mardm.append(motion_multimodality_batch_mardm)
        else:
            with torch.no_grad():
                sampled_tokens = model_without_ddp.sample_tokens(bsz=bs, m_lens=m_length, cfg=cfg,
                                                                 cfg_schedule=cfg_schedule, labels=clip_text,
                                                                 temperature=temperature, acc_ratio=acc_ratio)
                feats_rst = vae_model.decode(sampled_tokens / std_factor)

                feats_rst_denorm = datamodule.inv_transform_torch(feats_rst)
                feats_rst_mardm_eval = datamodule.feats_to_eval_mardm(feats_rst)

                et_tmr_pred, em_tmr_pred = tmr_wrapper.get_co_embeddings(feats_rst_denorm, m_length,
                                                                         clip_text=clip_text)
                (et_mardm_pred, em_mardm_pred), (
                clip_et_mardm_pred, clip_em_mardm_pred) = mardm_wrapper.get_co_embeddings(
                    word_embeddings, pos_one_hots, sent_len, clip_text, feats_rst_mardm_eval, m_length)

        feats_ref_denorm = datamodule.inv_transform_torch(feats_ref)
        feats_ref_mardm_eval = datamodule.feats_to_eval_mardm(feats_ref)

        et_tmr, em_tmr = tmr_wrapper.get_co_embeddings(feats_ref_denorm, m_length, clip_text=clip_text)
        (et_mardm, em_mardm), (clip_et_mardm, clip_em_mardm) = mardm_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, clip_text, feats_ref_mardm_eval, m_length)

        batch_clip_score_pred, batch_clip_score_real = 0, 0
        for j in range(32):
            single_em_pred = clip_em_mardm_pred[j]
            single_et_pred = clip_et_mardm_pred[j]
            clip_core = (single_em_pred @ single_et_pred.T).item()
            batch_clip_score_pred += clip_core

            single_em_real = clip_em_mardm[j]
            single_et_real = clip_et_mardm[j]
            clip_core_gt = (single_em_real @ single_et_real.T).item()
            batch_clip_score_real +=  clip_core_gt

        clip_score_real += batch_clip_score_real
        clip_score_pred += batch_clip_score_pred

        tmr_real_list.append(em_tmr)
        tmr_pred_list.append(em_tmr_pred)
        mardm_real_list.append(em_mardm)
        mardm_pred_list.append(em_mardm_pred)

        tmr_R_real = calculate_R_precision(et_tmr.cpu().numpy(), em_tmr.cpu().numpy(), top_k=3, sum_all=True)
        tmr_match_real = euclidean_distance_matrix(et_tmr.cpu().numpy(), em_tmr.cpu().numpy()).trace()
        RP_tmr_real += tmr_R_real
        matching_score_tmr_real += tmr_match_real

        tmr_R_pred = calculate_R_precision(et_tmr_pred.cpu().numpy(), em_tmr_pred.cpu().numpy(), top_k=3, sum_all=True)
        tmr_match_pred = euclidean_distance_matrix(et_tmr_pred.cpu().numpy(), em_tmr_pred.cpu().numpy()).trace()
        RP_tmr_pred += tmr_R_pred
        matching_score_tmr_pred += tmr_match_pred

        mardm_R_real = calculate_R_precision(et_mardm.cpu().numpy(), em_mardm.cpu().numpy(), top_k=3, sum_all=True)
        mardm_match_real = euclidean_distance_matrix(et_mardm.cpu().numpy(), em_mardm.cpu().numpy()).trace()
        RP_mardm_real += mardm_R_real
        matching_score_mardm_real += mardm_match_real

        mardm_R_pred = calculate_R_precision(et_mardm_pred.cpu().numpy(), em_mardm_pred.cpu().numpy(), top_k=3, sum_all=True)
        mardm_match_pred = euclidean_distance_matrix(et_mardm_pred.cpu().numpy(), em_mardm_pred.cpu().numpy()).trace()
        RP_mardm_pred += mardm_R_pred
        matching_score_mardm_pred += mardm_match_pred

        nb_sample += bs

    tmr_real_list_np = torch.cat(tmr_real_list, dim=0).cpu().numpy()
    tmr_pred_list_np = torch.cat(tmr_pred_list, dim=0).cpu().numpy()
    mardm_real_list_np = torch.cat(mardm_real_list, dim=0).cpu().numpy()
    mardm_pred_list_np = torch.cat(mardm_pred_list, dim=0).cpu().numpy()

    if cal_mm:
        motion_multimodality_tmr = torch.cat(motion_multimodality_tmr, dim=0).cpu().numpy()
        motion_multimodality_mardm = torch.cat(motion_multimodality_mardm, dim=0).cpu().numpy()

        multimodality_tmr = calculate_multimodality(motion_multimodality_tmr, 10)
        multimodality_mardm = calculate_multimodality(motion_multimodality_mardm, 10)

    mu_tmr_real, cov_tmr_real = calculate_activation_statistics_normalized(tmr_real_list_np)
    mu_tmr_pred, cov_tmr_pred = calculate_activation_statistics_normalized(tmr_pred_list_np)

    mu_mardm_real, cov_mardm_real = calculate_activation_statistics(mardm_real_list_np)
    mu_mardm_pred, cov_mardm_pred = calculate_activation_statistics(mardm_pred_list_np)

    fid_tmr = calculate_frechet_distance(mu_tmr_real, cov_tmr_real, mu_tmr_pred, cov_tmr_pred)
    fid_mardm = calculate_frechet_distance(mu_mardm_real, cov_mardm_real, mu_mardm_pred, cov_mardm_pred)

    RP_tmr_real = RP_tmr_real / nb_sample
    RP_tmr_pred = RP_tmr_pred / nb_sample
    matching_score_tmr_real = matching_score_tmr_real / nb_sample
    matching_score_tmr_pred = matching_score_tmr_pred / nb_sample

    RP_mardm_real = RP_mardm_real / nb_sample
    RP_mardm_pred = RP_mardm_pred / nb_sample
    matching_score_mardm_real = matching_score_mardm_real / nb_sample
    matching_score_mardm_pred = matching_score_mardm_pred / nb_sample
    clip_score_real = clip_score_real / nb_sample
    clip_score_pred = clip_score_pred / nb_sample

    msg_tmr = f"--> \t Ep {ep} :, FID_TMR. {fid_tmr:.4f}, RP_TMR_real. {RP_tmr_real}, RP_tmr_pred. {RP_tmr_pred}, matching_score_tmr_real. {matching_score_tmr_real}, matching_score_tmr_pred. {matching_score_tmr_pred}, multimodality_tmr {multimodality_tmr}"
    print(msg_tmr)
    msg_mardm = f"--> \t Ep {ep} :, FID_MARDM. {fid_mardm:.4f}, RP_mardm_real. {RP_mardm_real}, RP_mardm_pred. {RP_mardm_pred}, matching_score_mardm_real. {matching_score_mardm_real}, matching_score_mardm_pred. {matching_score_mardm_pred},  clip score real. {clip_score_real} clip score pred. {clip_score_pred}, multimodality_mardm {multimodality_mardm}"
    print(msg_mardm)

    return fid_tmr, RP_tmr_pred[0], RP_tmr_pred[1], RP_tmr_pred[2], matching_score_tmr_pred, fid_mardm, RP_mardm_pred[0], RP_mardm_pred[1], RP_mardm_pred[2], matching_score_mardm_pred, clip_score_pred, multimodality_tmr, multimodality_mardm



@torch.no_grad()
def eval_molingo_ms(val_loader, model_without_ddp, vae_model, ep, cfg,
                motionencoder, textencoder,
                temperature=1., std_factor=1., acc_ratio=1.):

    tmr_real_list, tmr_pred_list = [], []
    RP_tmr_real, RP_tmr_pred = 0, 0
    matching_score_tmr_real, matching_score_tmr_pred = 0, 0

    nb_sample = 0

    for i, batch in tqdm(enumerate(val_loader)):
        clip_text, feats_ref, m_length = batch
        feats_ref = feats_ref.detach().cuda().float()
        m_length = m_length.cuda()
        bs, token_len = feats_ref.shape[:2]
        with torch.no_grad():
            sampled_tokens = model_without_ddp.sample_tokens(bsz=bs, m_lens=m_length, cfg=cfg,
                                                             cfg_schedule="constant", labels=clip_text,
                                                             temperature=temperature, acc_ratio=acc_ratio)
            feats_rst = vae_model.decode(sampled_tokens / std_factor)

        et_tmr_pred, em_tmr_pred = textencoder(clip_text).loc, motionencoder(feats_rst, m_length).loc
        et_tmr, em_tmr = textencoder(clip_text).loc, motionencoder(feats_ref, m_length).loc

        tmr_real_list.append(em_tmr)
        tmr_pred_list.append(em_tmr_pred)

        tmr_R_real = calculate_R_precision(et_tmr.cpu().numpy(), em_tmr.cpu().numpy(), top_k=3, sum_all=True)
        tmr_match_real = euclidean_distance_matrix(et_tmr.cpu().numpy(), em_tmr.cpu().numpy()).trace()
        RP_tmr_real += tmr_R_real
        matching_score_tmr_real += tmr_match_real

        tmr_R_pred = calculate_R_precision(et_tmr_pred.cpu().numpy(), em_tmr_pred.cpu().numpy(), top_k=3, sum_all=True)
        tmr_match_pred = euclidean_distance_matrix(et_tmr_pred.cpu().numpy(), em_tmr_pred.cpu().numpy()).trace()
        RP_tmr_pred += tmr_R_pred
        matching_score_tmr_pred += tmr_match_pred

        nb_sample += bs

    tmr_real_list_np = torch.cat(tmr_real_list, dim=0).cpu().numpy()
    tmr_pred_list_np = torch.cat(tmr_pred_list, dim=0).cpu().numpy()

    mu_tmr_real, cov_tmr_real = calculate_activation_statistics(tmr_real_list_np)
    mu_tmr_pred, cov_tmr_pred = calculate_activation_statistics(tmr_pred_list_np)

    fid_tmr = calculate_frechet_distance(mu_tmr_real, cov_tmr_real, mu_tmr_pred, cov_tmr_pred)

    RP_tmr_real = RP_tmr_real / nb_sample
    RP_tmr_pred = RP_tmr_pred / nb_sample

    matching_score_tmr_pred = matching_score_tmr_pred / nb_sample

    msg_tmr = f"--> \t Ep {ep} :, FID_TMR. {fid_tmr:.4f}, RP_TMR_real. {RP_tmr_real}, RP_tmr_pred. {RP_tmr_pred}, matching_score_tmr_pred. {matching_score_tmr_pred}"
    print(msg_tmr)
    return fid_tmr, RP_tmr_pred[0], RP_tmr_pred[1], RP_tmr_pred[2], matching_score_tmr_pred


@torch.no_grad()
def eval_vae_ms(test_loader, vae_model, ms_wrapper, datamodule, ae, device='cuda'):
    vae_model.eval()

    em_ref_list, em_rst_list = [], []
    joints_ref_list = []
    joints_rst_list = []

    for batch_data in test_loader:
        _, feats_ref, _, _, lengths = batch_data
        feats_ref = feats_ref.detach().to(device).float()
        lengths = lengths.to(device)
        if not ae:
            feats_rst, z, _ = vae_model(feats_ref)
        else:
            feats_rst, z = vae_model(feats_ref)

        joints_ref = datamodule.feats2joints(feats_ref)
        joints_rst = datamodule.feats2joints(feats_rst)

        bs = joints_ref.shape[0]
        valid_joints_ref, valid_joints_rst = [], []
        for b in range(bs):
            valid_joints_ref.append(joints_ref[b][:lengths[b].item()])
            valid_joints_rst.append(joints_rst[b][:lengths[b].item()])

        valid_joints_ref = torch.cat(valid_joints_ref, dim=0)
        valid_joints_rst = torch.cat(valid_joints_rst, dim=0)

        joints_ref_list.append(valid_joints_ref)
        joints_rst_list.append(valid_joints_rst)

        em_ref = ms_wrapper(feats_ref, lengths).loc
        em_rst = ms_wrapper(feats_rst, lengths).loc

        em_ref_list.append(em_ref)
        em_rst_list.append(em_rst)

    em_ref_np = torch.cat(em_ref_list, dim=0).cpu().numpy()
    em_rst_np = torch.cat(em_rst_list, dim=0).cpu().numpy()

    joints_ref_np = torch.cat(joints_ref_list, dim=0).detach()
    joints_rst_np = torch.cat(joints_rst_list, dim=0).detach()

    tmr_ref_mu, tmr_ref_cov = calculate_activation_statistics(em_ref_np)
    tmr_rst_mu, tmr_rst_cov = calculate_activation_statistics(em_rst_np)

    fid = calculate_frechet_distance(tmr_ref_mu, tmr_ref_cov, tmr_rst_mu, tmr_rst_cov)
    mpjpe = calculate_mpjpe(joints_ref_np, joints_rst_np).mean().item() * 1000

    vae_model.train()

    return mpjpe, fid


@torch.no_grad()
def eval_molingo_during_training(out_dir, val_loader, model_without_ddp, vae_model, datamodule, ema_params, ep, cfg,
                                 temperature, best_fid_tmr, best_top1_tmr, best_top2_tmr, best_top3_tmr,
                                 best_matching_score_tmr, motionencoder, textencoder, plot_func, rank=0,
                                 save_ckpt=False, save_anim=False, use_ema=True, std_factor=1., acc_ratio=1.):
    out_model_dir = pjoin(out_dir, 'model')
    model_without_ddp.eval()
    if use_ema:
        model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
            assert name in ema_state_dict
            ema_state_dict[name] = ema_params[i]
        print("Switch to ema")
        model_without_ddp.load_state_dict(ema_state_dict)
    torch.cuda.synchronize()

    tmr_real_list, tmr_pred_list = [], []
    RP_tmr_real, RP_tmr_pred = 0, 0
    matching_score_tmr_real, matching_score_tmr_pred = 0, 0
    nb_sample = 0

    for i, batch in tqdm(enumerate(val_loader)):
        clip_text, feats_ref, m_length = batch
        feats_ref = feats_ref.detach().cuda().float()
        m_length = m_length.cuda()
        bs, token_len = feats_ref.shape[:2]
        with torch.no_grad():
            sampled_tokens = model_without_ddp.sample_tokens(bsz=bs, m_lens=m_length, cfg=cfg,
                                                             cfg_schedule="linear", labels=clip_text,
                                                             temperature=temperature, acc_ratio=acc_ratio)
            feats_rst = vae_model.decode(sampled_tokens / std_factor)

        et_tmr_pred, em_tmr_pred = textencoder(clip_text).loc, motionencoder(feats_rst, m_length).loc
        et_tmr, em_tmr = textencoder(clip_text).loc, motionencoder(feats_ref, m_length).loc

        tmr_real_list.append(em_tmr)
        tmr_pred_list.append(em_tmr_pred)

        tmr_R_real = calculate_R_precision(et_tmr.cpu().numpy(), em_tmr.cpu().numpy(), top_k=3, sum_all=True)
        tmr_match_real = euclidean_distance_matrix(et_tmr.cpu().numpy(), em_tmr.cpu().numpy()).trace()
        RP_tmr_real += tmr_R_real
        matching_score_tmr_real += tmr_match_real

        tmr_R_pred = calculate_R_precision(et_tmr_pred.cpu().numpy(), em_tmr_pred.cpu().numpy(), top_k=3, sum_all=True)
        tmr_match_pred = euclidean_distance_matrix(et_tmr_pred.cpu().numpy(), em_tmr_pred.cpu().numpy()).trace()
        RP_tmr_pred += tmr_R_pred
        matching_score_tmr_pred += tmr_match_pred

        nb_sample += bs

    tmr_real_list_np = torch.cat(tmr_real_list, dim=0).cpu().numpy()
    tmr_pred_list_np = torch.cat(tmr_pred_list, dim=0).cpu().numpy()

    mu_tmr_real, cov_tmr_real = calculate_activation_statistics(tmr_real_list_np)
    mu_tmr_pred, cov_tmr_pred = calculate_activation_statistics(tmr_pred_list_np)

    fid_tmr = calculate_frechet_distance(mu_tmr_real, cov_tmr_real, mu_tmr_pred, cov_tmr_pred)

    RP_tmr_real = RP_tmr_real / nb_sample
    RP_tmr_pred = RP_tmr_pred / nb_sample

    matching_score_tmr_real = matching_score_tmr_real / nb_sample
    matching_score_tmr_pred = matching_score_tmr_pred / nb_sample

    if rank == 0:
        wandb.log({
            "tmr/fid": fid_tmr,
            "tmr/r1": RP_tmr_pred[0],
            "tmr/r2": RP_tmr_pred[1],
            "tmr/r3": RP_tmr_pred[2],
            "tmr/mscore": matching_score_tmr_pred,
        }, step=ep)

    msg_tmr = (f"--> \t Ep {ep} :, FID_TMR. {fid_tmr:.4f}, RP_TMR_real. {RP_tmr_real}, "
               f"RP_tmr_pred. {RP_tmr_pred}, matching_score_tmr_real. {matching_score_tmr_real}, "
               f"matching_score_tmr_pred. {matching_score_tmr_pred}")
    print(msg_tmr)

    if fid_tmr < best_fid_tmr:
        print(f"--> --> \t FID_TMR Improved from {best_fid_tmr:.5f} to {fid_tmr:.5f} !!!")
        best_fid_tmr = fid_tmr
        if save_ckpt:
            misc.save_model_simplified(out_model_dir, model_without_ddp=model_without_ddp, epoch=ep,
                                       ema_params=ema_params, epoch_name="net_best_fid_tmr")

    if matching_score_tmr_pred < best_matching_score_tmr:
        print(f"--> --> \t matching_score TMR Improved from {best_matching_score_tmr:.5f} "
              f"to {matching_score_tmr_pred:.5f} !!!")
        best_matching_score_tmr = matching_score_tmr_pred

    if RP_tmr_pred[0] > best_top1_tmr:
        print(f"--> --> \t Top1 TMR Improved from {best_top1_tmr:.4f} to {RP_tmr_pred[0]:.4f} !!!")
        best_top1_tmr = RP_tmr_pred[0]

    if RP_tmr_pred[1] > best_top2_tmr:
        print(f"--> --> \t Top2 TMR Improved from {best_top2_tmr:.4f} to {RP_tmr_pred[1]:.4f} !!!")
        best_top2_tmr = RP_tmr_pred[1]

    if RP_tmr_pred[2] > best_top3_tmr:
        print(f"--> --> \t Top3 TMR Improved from {best_top3_tmr:.4f} to {RP_tmr_pred[2]:.4f} !!!")
        best_top3_tmr = RP_tmr_pred[2]

    if save_anim:
        rand_idx = torch.randint(bs, (16,))
        data = feats_rst[rand_idx].detach().cpu().numpy()
        captions = [clip_text[k] for k in rand_idx]
        lengths = m_length[rand_idx].cpu().numpy()
        save_dir = pjoin(out_dir, 'animation', 'E%04d' % ep)
        os.makedirs(save_dir, exist_ok=True)
        plot_func(data, save_dir, captions, lengths)

    if use_ema:
        print("Switch back from ema")
        model_without_ddp.load_state_dict(model_state_dict)

    return best_fid_tmr, best_top1_tmr, best_top2_tmr, best_top3_tmr, best_matching_score_tmr