import os
import shutil
from os.path import join as pjoin
import random
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from typing import Optional

from diffusers.optimization import get_scheduler

from mogen.core.eval import eval_vae_ms
from mogen.utils.plot_script import plot_single_motion

target_low, target_high = 3.0, 7.0


def def_value():
    return 0.0


def lengths_to_mask(lengths, max_len, device=None):
    lengths = torch.as_tensor(lengths, device=device)
    return torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)


class SAETrainer:
    def __init__(self, args, vae_model, datamodule, ms_wrapper):
        self.opt = args
        self.vae_model = vae_model
        self.datamodule = datamodule
        self.ms_wrapper = ms_wrapper
        self.device = args.device

        if args.is_train:
            if args.recons_loss == 'l1':
                self.l1_criterion = torch.nn.L1Loss()
            elif args.recons_loss == 'l1_smooth':
                self.l1_criterion = torch.nn.SmoothL1Loss()

        self.feats2joints = datamodule.feats2joints
        self.mask_loss = False

        self.rec_feats_loss = args.rec_feats_loss
        self.rec_joints_loss = args.rec_joints_loss
        self.rec_velocity_loss = args.rec_velocity_loss
        self.rec_root_loss = args.rec_root_loss

        self.rec_feats_ratio = args.rec_feats_ratio
        self.rec_joints_ratio = args.rec_joints_ratio
        self.rec_velocity_ratio = args.rec_velocity_ratio
        self.rec_root_ratio = args.rec_root_ratio
        self.kl_ratio = args.kl_ratio if not self.opt.ae else 0.0
        self.cosine_ratio = args.cosine_ratio
        self.filter = args.filter

        self.njoints = 22

        self.t5_proj = nn.Linear(1024, args.output_emb_width).to(self.device)

        wandb.init(
            project='sae-molingo',
            name=self.opt.name,
        )

    def loss_calculate(self, a: torch.Tensor, b: torch.Tensor, loss_type: str,
                       mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        mask = None if not self.mask_loss else mask
        if loss_type == 'l1':
            loss = F.l1_loss(a, b, reduction='none')
        elif loss_type == 'l1_smooth':
            loss = F.smooth_l1_loss(a, b, reduction='none')
        elif loss_type == 'l2':
            loss = F.mse_loss(a, b, reduction='none')
        else:
            raise ValueError(f'Unknown loss type: {loss_type}')

        if mask is not None:
            loss = (loss.mean(dim=-1) * mask).sum(-1) / mask.sum(-1)
        return loss.mean()

    def visualize(self, step, batch_data, num_vis=8):

        step_str = str(step).zfill(9)
        save_dir = pjoin(self.opt.ani_dir, f'step_{step_str}')
        os.makedirs(save_dir, exist_ok=True)

        _, feats_ref, _, _, lengths = batch_data
        feats_ref = feats_ref.detach().to(self.device).float()
        if not self.opt.ae:
            feats_rst, z, dist_m = self.vae_model(feats_ref)
        else:
            feats_rst, z = self.vae_model(feats_ref)

        joints_ref = self.datamodule.feats2joints(feats_ref).detach().cpu().numpy()  # [bs, T, 22, 3]
        joints_rst = self.datamodule.feats2joints(feats_rst).detach().cpu().numpy()

        bs = joints_ref.shape[0]
        indices = random.sample(range(bs), num_vis)

        joints_ref = joints_ref[indices]
        joints_rst = joints_rst[indices]

        for k in range(num_vis):
            cur_joints_ref, cur_joints_rst = joints_ref[k], joints_rst[k]
            k_str = str(k).zfill(4)
            video_path_ref = pjoin(save_dir, f'{k_str}_in.mp4')
            video_path_rst = pjoin(save_dir, f'{k_str}_out.mp4')
            try:
                plot_single_motion(cur_joints_ref, video_path_ref, fps=30)
                plot_single_motion(cur_joints_rst, video_path_rst, fps=30)
            except:
                pass

    def compute_cosine_loss(self, z, t5_vec, has_babel, lengths):
        """Cosine-similarity supervision between VAE latent tokens and per-frame
        T5 babel embeddings aggregated to the latent-token rate.

        For each latent token `ii` (stride = 2 ** down_t), we mean-pool T5 frame
        embeddings over a causal history window of `4 * stride` frames plus the
        `stride` frames the latent "owns", then project 1024-d T5 into the latent
        dim via `self.t5_proj` and compute `1 - cos(z, projected_t5)` over
        unpadded tokens of babel-annotated samples. Returns a scalar (0 if no
        sample in the batch has babel).
        """
        if not has_babel.any():
            return torch.tensor(0., device=z.device)

        # Aggregate per-frame T5 → per-latent-token embeddings, then project.
        # Window scales with stride so the history stays proportional to the
        # downsample rate (16 frames at down_t=2, 8 at down_t=1, etc.).
        bs, token_len, _ = z.shape
        stride = 2 ** self.opt.down_t
        history = 4 * stride
        t5_agg = torch.zeros(bs, token_len, t5_vec.shape[-1], device=z.device, dtype=z.dtype)
        for ii in range(token_len):
            start = max(stride * ii - history, 0)
            end = stride * ii + stride  # Python-exclusive upper bound
            t5_agg[:, ii, :] = t5_vec[:, start:end, :].mean(dim=-2)
        t5_embed = self.t5_proj(t5_agg)

        # Restrict to babel-annotated samples and unpadded latent tokens.
        latent_mask = lengths_to_mask(lengths // self.opt.unit_length, token_len).to(z.device)
        z_flat = z[has_babel].reshape(-1, z.shape[-1])
        t_flat = t5_embed[has_babel].reshape(-1, t5_embed.shape[-1])
        valid = latent_mask[has_babel].reshape(-1)
        z_flat = z_flat[valid]
        t_flat = t_flat[valid]

        # Drop near-duplicate consecutive text tokens to prevent trivial collapse.
        if self.filter and z_flat.shape[0] > 1:
            adj_sim = (t_flat[1:] * t_flat[:-1]).sum(dim=-1)
            keep = torch.zeros(z_flat.shape[0], dtype=torch.bool, device=z.device)
            keep[0] = True
            keep[1:] = adj_sim < 0.995
            z_flat = z_flat[keep]
            t_flat = t_flat[keep]

        return (1.0 - F.cosine_similarity(z_flat, t_flat, dim=-1, eps=1e-8)).mean()

    def train_vae_forward(self, batch_data):
        _, feats_ref, t5_vec, has_babel, lengths = batch_data
        feats_ref = feats_ref.detach().to(self.device).float()
        t5_vec = t5_vec.detach().to(self.device).float()
        has_babel = has_babel.to(self.device)

        if not self.opt.ae:
            feats_rst, z, dist_m = self.vae_model(feats_ref)
        else:
            feats_rst, z = self.vae_model(feats_ref)

        loss_dict = dict(
            rec_feats_loss=torch.tensor(0., device=z.device),
            rec_joints_loss=torch.tensor(0., device=z.device),
            rec_velocity_loss=torch.tensor(0., device=z.device),
            rec_root_loss=torch.tensor(0., device=z.device),
            kl_loss=torch.tensor(0., device=z.device),
            cosine_loss=torch.tensor(0., device=z.device))

        has_pair = has_babel.any()
        if has_pair:
            cosine_loss = self.compute_cosine_loss(z, t5_vec, has_babel, lengths)
            loss_dict['cosine_loss'] = cosine_loss * self.cosine_ratio

        if self.rec_feats_ratio > 0:
            rec_feats_loss = self.loss_calculate(feats_ref, feats_rst, self.rec_feats_loss)
            loss_dict['rec_feats_loss'] = rec_feats_loss * self.rec_feats_ratio

        if self.rec_joints_ratio > 0:
            joints_rst = self.feats2joints(feats_rst).reshape(feats_ref.size(0), feats_ref.size(1), -1)
            joints_ref = self.feats2joints(feats_ref).reshape(feats_ref.size(0), feats_ref.size(1), -1)
            rec_joints_loss = self.loss_calculate(joints_ref, joints_rst, self.rec_joints_loss)
            loss_dict['rec_joints_loss'] = rec_joints_loss * self.rec_joints_ratio

        if self.rec_velocity_ratio > 0:
            assert self.rec_joints_ratio > 0
            vel_rst = joints_rst[:, 1:] - joints_rst[:, :-1]
            vel_ref = joints_ref[:, 1:] - joints_ref[:, :-1]
            rec_velocity_loss = self.loss_calculate(vel_ref, vel_rst, self.rec_velocity_loss)

            loss_dict['rec_velocity_loss'] = rec_velocity_loss * self.rec_velocity_ratio

        if self.rec_root_ratio > 0:
            rec_root_loss = self.loss_calculate(feats_ref[..., :8], feats_rst[..., :8], self.rec_root_loss)
            loss_dict['rec_root_loss'] = rec_root_loss * self.rec_root_ratio

        if self.kl_ratio > 0:
            mu_ref = torch.zeros_like(dist_m.loc)
            scale_ref = torch.ones_like(dist_m.scale)
            dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
            kl_loss = torch.distributions.kl_divergence(dist_m, dist_ref).mean()
            loss_dict['kl_loss'] = kl_loss * self.kl_ratio

        loss = sum([v for v in loss_dict.values()])
        loss_dict['loss'] = loss
        return loss_dict, has_pair

    def train(self, train_loader, test_loader):
        self.vae_model.to(self.device)

        max_train_steps = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {max_train_steps}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(test_loader)))

        self.optimizer = torch.optim.AdamW(
            list(self.vae_model.parameters()) + list(self.t5_proj.parameters()),
            lr=self.opt.lr,
            betas=(self.opt.adam_beta1, self.opt.adam_beta2),
            weight_decay=self.opt.adam_weight_decay,
            eps=self.opt.adam_epsilon)

        self.lr_scheduler = get_scheduler(
            self.opt.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.opt.warm_up_iter,
            num_training_steps=max_train_steps)

        # Train!
        print("***** Running training *****")
        print(f"  Num examples = {len(train_loader)}")
        print(f"  Num Epochs = {self.opt.max_epoch}")
        print(f"  Instantaneous batch size per device = {self.opt.batch_size}")
        print(f"  Total optimization steps = {max_train_steps}")

        epoch = 0
        global_step = 0

        resume_path = pjoin(self.opt.model_dir, "checkpoint-last.ckpt")
        if os.path.exists(resume_path):
            print(f'Resuming from {resume_path}.....')
            state_dict = torch.load(resume_path, map_location="cpu")["state_dict"]
            self.vae_model.load_state_dict(state_dict=state_dict)
            print(f'Successfully resumed from {resume_path} !!!')

        # Eval first round
        min_mpjpe, min_fid = eval_vae_ms(test_loader, self.vae_model, self.ms_wrapper, self.datamodule, self.opt.ae, self.device)

        print(f'initial mpjpe {min_mpjpe} FID {min_fid}')

        progress_bar = tqdm(range(0, max_train_steps), desc="Steps")
        checkpointing_steps = self.opt.save_every_e * len(train_loader)
        validation_steps = self.opt.eval_every_e * len(train_loader)
        animation_steps = self.opt.anim_every_e * len(train_loader)

        print(f'checkpointing steps {checkpointing_steps}')
        print(f'validation steps {validation_steps}')
        print(f'animation steps {animation_steps}')

        rec_feats_loss = 0.0
        rec_joints_loss = 0.0
        rec_velocity_loss = 0.0
        rec_root_loss = 0.0
        kl_loss = 0.0
        cosine_loss = 0.0
        loss = 0.0

        while epoch < self.opt.max_epoch:
            self.vae_model.train()
            has_pair_glob = False

            for i, batch_data in enumerate(train_loader):
                loss_dict, has_pair = self.train_vae_forward(batch_data)
                has_pair_glob = has_pair_glob | bool(has_pair)

                rec_feats_loss = loss_dict['rec_feats_loss']
                rec_joints_loss = loss_dict['rec_joints_loss']
                rec_velocity_loss = loss_dict['rec_velocity_loss']
                rec_root_loss = loss_dict['rec_root_loss']
                kl_loss = loss_dict['kl_loss']
                loss = loss_dict['loss']
                if has_pair:
                    cosine_loss = loss_dict['cosine_loss']

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.vae_model.parameters()) + list(self.t5_proj.parameters()),
                    self.opt.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                global_step += 1

                if global_step % animation_steps == 0:
                    self.visualize(global_step, batch_data)

                if global_step % validation_steps == 0:
                    cur_mpjpe, cur_fid = eval_vae_ms(test_loader, self.vae_model, self.ms_wrapper, self.datamodule, self.opt.ae,
                                                                     self.device)

                    wandb.log({
                        "eval/mpjpe": cur_mpjpe,
                        "eval/fid": cur_fid,
                    }, step=global_step)

                    if cur_mpjpe < min_mpjpe:
                        min_mpjpe = cur_mpjpe
                        save_path = os.path.join(self.opt.model_dir,
                                                 f"checkpoint-{global_step}-mpjpe-{round(cur_mpjpe, 3)}-fid_tmr-{round(cur_fid, 3)}.ckpt")
                        ckpt = dict(state_dict=self.vae_model.state_dict())
                        self.vae_model.on_save_checkpoint(ckpt)
                        torch.save(ckpt, save_path)
                        print(f"Saved state to {save_path} with mpjpe: {round(cur_mpjpe, 3)}")

                    if cur_fid < min_fid:
                        min_fid = cur_fid
                        save_path = os.path.join(self.opt.model_dir,
                                                 f"checkpoint-{global_step}-fid_tmr-{round(cur_fid, 3)}-mpjpe-{round(cur_mpjpe, 3)}.ckpt")
                        ckpt = dict(state_dict=self.vae_model.state_dict())
                        self.vae_model.on_save_checkpoint(ckpt)
                        torch.save(ckpt, save_path)
                        print(f"Saved state to {save_path} with fid: {round(cur_fid, 3)}")

                        best_fid_path = os.path.join(self.opt.model_dir, "net_best_fid.ckpt")
                        if os.path.exists(best_fid_path):
                            os.remove(best_fid_path)
                        shutil.copyfile(save_path, best_fid_path)


                logs = {"loss": loss.item(),
                        "lr": self.lr_scheduler.get_last_lr()[0],
                        "rec_feats_loss": rec_feats_loss.item(),
                        'rec_joints_loss': rec_joints_loss.item(),
                        'rec_velocity_loss': rec_velocity_loss.item(),
                        'kl_loss': kl_loss.item()}
                if has_pair:
                    logs['cosine_loss'] = cosine_loss.item()

                if global_step % 15000 == 0:
                    progress_bar.set_postfix(**logs)

                if global_step >= max_train_steps:
                    save_path = os.path.join(self.opt.model_dir, "checkpoint-last.ckpt")
                    ckpt = dict(state_dict=self.vae_model.state_dict())
                    self.vae_model.on_save_checkpoint(ckpt)
                    torch.save(ckpt, save_path)
                    exit(0)

            wandb.log({
                "train/rec_feats_loss": rec_feats_loss,
                "train/rec_joints_loss": rec_joints_loss,
                "train/rec_velocity_loss": rec_velocity_loss,
                "train/kl_loss": kl_loss,
                "train/loss": loss.item(),
            }, step=global_step)

            if has_pair_glob:
                wandb.log({
                    "train/cosine_loss": cosine_loss,
                }, step=global_step)

            epoch += 1