import argparse
import os
import torch
import math

def arg_parse(is_train=False):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## dataloader
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--dataset_name', type=str, default='ms', help='dataset directory')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument("--gpu_id", type=int, default=0, help='GPU id')
    parser.add_argument("--exp_name", type=str, default='ms', help='short tag baked into the run name')
    parser.add_argument("--data_root", type=str, default='/home/hynann/data', help='Root folder for training data')

    ## optimization
    parser.add_argument('--max_epoch', default=5000, type=int, help='number of total epochs to run')
    parser.add_argument('--warm_up_iter', default=1000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=5e-5, type=float, help='max learning rate')
    parser.add_argument("--lr_scheduler", type=str, default='cosine', help='lr scheduler')

    parser.add_argument('--adam_weight_decay', default=0.0, type=float, help='weight decay')
    parser.add_argument('--recons_loss', type=str, default='l1_smooth', help='reconstruction loss')
    parser.add_argument('--adam_beta1', default=0.9, type=float, help='adam beta1')
    parser.add_argument('--adam_beta2', default=0.999, type=float, help='adam beta2')
    parser.add_argument('--adam_epsilon', default=1e-08, type=float, help='adam epsilon')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max grad norm')

    ## loss
    parser.add_argument('--rec_feats_ratio', default=1.0, type=float, help='rec feats ratio')
    parser.add_argument('--rec_joints_ratio', default=1.0, type=float, help='rec joints ratio')
    parser.add_argument('--rec_root_ratio', default=0.0, type=float, help='rec root ratio')
    parser.add_argument('--rec_velocity_ratio', default=10.0, type=float, help='rec velocity ratio')
    parser.add_argument('--kl_ratio', default=1e-5, type=float, help='kl_ratio')
    parser.add_argument('--cosine_ratio', default=0.001, type=float, help='weight for the cosine-similarity supervision loss')

    parser.add_argument('--loss_type', type=str, default='l2', help='reconstruction loss variant (l1, l1_smooth, l2)')
    parser.add_argument('--pad_mode', type=str, default='zero', help='conv pad mode for encoder/decoder')

    ## vae arch
    parser.add_argument("--down_t", type=int, default=1, help="downsampling rate")
    parser.add_argument("--stride_t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=1024, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="num of resblocks for each res")
    parser.add_argument("--dilation_growth_rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output_emb_width", type=int, default=32, help="output embedding width")
    parser.add_argument('--activation', type=str, default='silu', choices=['relu', 'silu', 'gelu'],
                        help='encoder/decoder activation')
    parser.add_argument('--norm', type=str, default=None, help='encoder/decoder norm')

    ## other
    parser.add_argument('--name', type=str, default="test", help='Name of this trial')
    parser.add_argument('--checkpoints_dir', type=str, default='mogen/checkpoints', help='models are saved here')
    parser.add_argument('--save_every_e', default=50, type=int, help='save model every n epoch')
    parser.add_argument('--eval_every_e', default=50, type=int, help='save eval results every n epoch')
    parser.add_argument('--anim_every_e', default=125, type=int, help='save animation results every n epoch')

    parser.add_argument("--max_motion_length", type=int, default=300, help="Max length of motion")
    parser.add_argument("--unit_length", type=int, default=2, help="Downscale ratio of VQ")

    parser.add_argument("--seed", default=3407, type=int)
    parser.add_argument('--ae', action='store_true', help='train a plain AE (no KL) instead of a VAE')
    parser.add_argument('--filter', action='store_true', help='drop near-duplicate consecutive text tokens in cosine loss')

    opt = parser.parse_args()
    torch.cuda.set_device(opt.gpu_id)

    args = vars(opt)
    downsample_rate = int(math.pow(2, opt.down_t))
    opt.name = f'sae_{opt.exp_name}_{opt.loss_type}_{downsample_rate}_{opt.output_emb_width}_{opt.width}_d{opt.depth}_kl_{opt.kl_ratio}_zero_cos_{opt.cosine_ratio}'

    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    opt.is_train = is_train

    loss_type = opt.loss_type
    opt.rec_feats_loss = loss_type
    opt.rec_joints_loss = loss_type
    opt.rec_velocity_loss = loss_type
    opt.rec_root_loss = loss_type

    if is_train:
        expr_dir = os.path.join(opt.checkpoints_dir, opt.dataset_name, opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir, exist_ok=True)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
    return opt
