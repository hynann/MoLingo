import argparse
import os
import torch


def arg_parse(is_train=False):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## dataloader / data
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--dataset_name', type=str, default='ms', help='dataset directory')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--data_root', type=str, default='/home/hynann/data', help='root folder for training data')
    parser.add_argument('--exp_name', type=str, default='ms', help='short tag baked into the run name')

    ## distributed
    parser.add_argument('--world_size', type=int, default=1, help='number of distributed processes')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', type=str, default='env://', help='url used to set up distributed training')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id (single-GPU fallback)')

    ## optimisation
    parser.add_argument('--max_epoch', type=int, default=100000, help='maximum number of epochs for training')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--base_lr', type=float, default=5e-5, help='base learning rate')
    parser.add_argument('--warm_up_iter', type=int, default=2000, help='number of total iterations for warmup')
    parser.add_argument('--lr_schedule', type=str, default='cosine', help='lr scheduler feeding adjust_learning_rate')
    parser.add_argument('--seed', type=int, default=3407)

    ## model arch (only kwargs accepted by MoLingo.__init__)
    parser.add_argument('--model_size', type=str, default='large',
                        help='picks molingo_{tiny,base,large,huge} factory')
    parser.add_argument('--label_drop_prob', type=float, default=0.1, help='classifier-free-guidance drop prob')
    parser.add_argument('--proj_dropout', type=float, default=0.1)
    parser.add_argument('--flow_batch_mul', type=int, default=4)
    parser.add_argument('--sample_steps', type=int, default=32)
    parser.add_argument('--adapter_layers', type=int, default=6)
    parser.add_argument('--t5_max_len', type=int, default=64)
    parser.add_argument('--grad_checkpointing', action='store_true')

    ## VAE hookup / training
    parser.add_argument('--vae', type=str, required=False, default='sae_ms_l2_2_32_1024_d3_kl_1e-05_zero_cos_0.001',
                        help='SAE checkpoint folder name under checkpoints/<dataset>/')
    parser.add_argument('--checkpoints_dir', type=str, default='mogen/checkpoints', help='models are saved here')
    parser.add_argument('--std_factor', type=float, default=1.0)
    parser.add_argument('--cfg', type=float, default=4.0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--acc_ratio', type=float, default=5.0)

    ## eval / saving
    parser.add_argument('--eval_every_e', type=int, default=100)
    parser.add_argument('--save_every_e', type=int, default=150)

    opt = parser.parse_args()

    if opt.gpu_id != -1:
        torch.cuda.set_device(opt.gpu_id)

    args = vars(opt)
    opt.name = f'molingo_{opt.exp_name}_{opt.model_size}_vae_{opt.vae}'

    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    opt.is_train = is_train

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
