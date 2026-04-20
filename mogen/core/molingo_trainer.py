import sys
sys.path.append('.')

import copy
import math
import os
from os.path import join as pjoin

import torch
import torch.distributed as dist
import wandb

import mogen.utils.misc as misc
import mogen.utils.lr_sched as lr_sched
from mogen.core.eval import eval_molingo_during_training
from mogen.utils.misc import NativeScalerWithGradNormCount as NativeScaler


def is_main():
    return (not dist.is_available() or not dist.is_initialized()
            or dist.get_rank() == 0)


def def_value():
    return 0.0


def lengths_to_mask(lengths, device, max_len) -> torch.Tensor:
    lengths = torch.tensor(lengths, device=device)
    max_len = max_len if max_len else max(lengths)
    mask = torch.arange(max_len, device=device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


class MoLingoTrainer:
    def __init__(self, args, model, model_without_ddp, vae_model, datamodule, rank=0, ae=False):
        self.opt = args
        self.model = model
        self.ae = ae
        self.model_without_ddp = model_without_ddp
        self.vae_model = vae_model
        self.device = args.device
        self.datamodule = datamodule
        self.distributed = args.distributed
        self.vae_model.eval()
        self.rank = rank

        if rank == 0:
            wandb.init(
                project='molingo-ms',
                name=self.opt.name,
            )

    def forward(self, batch_data, plot_t2m, std_factor):
        conds, motion, m_lens = batch_data

        motion = motion.detach().float().to(self.device)

        with torch.no_grad():
            if not self.ae:
                x, _dist = self.vae_model.encode(motion)
            else:
                x = self.vae_model.ae_encode(motion)
            x = x.mul_(std_factor)

        with torch.cuda.amp.autocast():
            loss, loss_dict = self.model(x, conds, m_lens)

        return loss, loss_dict

    def train(self, train_loader, eval_val_loader, motionencoder, textencoder, plot_eval):
        self.model.to(self.device)
        self.vae_model.to(self.device)

        model_without_ddp = self.model_without_ddp
        loss_scaler = NativeScaler()
        base_lr = self.opt.base_lr

        param_groups = misc.add_weight_decay(model_without_ddp, 0.02)
        self.optimizer = torch.optim.AdamW(param_groups, lr=base_lr, betas=(0.9, 0.95))

        epoch = 0
        it = 0

        resume_path = pjoin(self.opt.model_dir, "net_best_fid.pth")
        if os.path.exists(resume_path):
            torch.cuda.empty_cache()
            checkpoint = torch.load(resume_path, map_location='cpu')
            print(f"Loading from resume path {resume_path}")
            model_without_ddp.load_state_dict(checkpoint['model'])
            model_params = list(model_without_ddp.parameters())
            ema_state_dict = checkpoint['model_ema']
            ema_params = [ema_state_dict[name].cuda() for name, _ in model_without_ddp.named_parameters()]
            print("Resume checkpoint %s" % resume_path)
            torch.cuda.empty_cache()
            del checkpoint
        else:
            model_params = list(model_without_ddp.parameters())
            ema_params = copy.deepcopy(model_params)
            print("Training from scratch")

        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(eval_val_loader)))

        best_fid_tmr = 5000.
        best_top1_tmr, best_top2_tmr, best_top3_tmr, best_matching_score_tmr = 0., 0., 0., 100.

        while epoch < self.opt.max_epoch:
            if self.opt.distributed:
                train_loader.sampler.set_epoch(epoch)
            self.model.train()
            torch.backends.cudnn.deterministic = False
            self.vae_model.eval()

            metric_logger = misc.MetricLogger(delimiter="  ")
            metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

            self.optimizer.zero_grad()

            loss_value = 0.0
            for i, batch in enumerate(train_loader):
                it += 1
                lr_sched.adjust_learning_rate(self.optimizer, i / len(train_loader) + epoch, base_lr, self.opt)

                loss, loss_dict = self.forward(batch, plot_eval, std_factor=self.opt.std_factor)
                loss_value = loss.item()

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    sys.exit(1)

                loss_scaler(loss, self.optimizer, clip_grad=3.0, parameters=self.model.parameters(), update_grad=True)
                self.optimizer.zero_grad()

                torch.cuda.synchronize()

                update_ema(ema_params, model_params, rate=0.9999)

                metric_logger.update(loss=loss_value)
                lr = self.optimizer.param_groups[0]["lr"]
                metric_logger.update(lr=lr)

            if is_main():
                wandb.log({"train/loss": loss_value}, step=epoch)

            metric_logger.synchronize_between_processes()
            print(f'epoch {epoch}  Averaged stats: {metric_logger}')

            epoch += 1

            if epoch >= 350 and epoch % self.opt.eval_every_e == 0:
                best_fid_tmr, best_top1_tmr, best_top2_tmr, best_top3_tmr, best_matching_score_tmr = eval_molingo_during_training(
                    self.opt.save_root, eval_val_loader, model_without_ddp, self.vae_model,
                    self.datamodule, ema_params, epoch,
                    cfg=self.opt.cfg, temperature=self.opt.temperature,
                    best_fid_tmr=best_fid_tmr,
                    best_top1_tmr=best_top1_tmr, best_top2_tmr=best_top2_tmr, best_top3_tmr=best_top3_tmr,
                    best_matching_score_tmr=best_matching_score_tmr,
                    motionencoder=motionencoder, textencoder=textencoder,
                    plot_func=plot_eval, rank=self.rank, save_ckpt=True, save_anim=False,
                    std_factor=self.opt.std_factor,
                    acc_ratio=self.opt.acc_ratio,
                )

            if epoch >= 350 and epoch % self.opt.save_every_e == 0:
                misc.save_model(self.opt.model_dir, model_without_ddp=model_without_ddp, optimizer=self.optimizer,
                                loss_scaler=loss_scaler, epoch=epoch, ema_params=ema_params, epoch_name="latest")


def update_ema(target_params, source_params, rate=0.99):
    """Update target parameters to be closer to source parameters via EMA."""
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)
