import math


def adjust_learning_rate(optimizer, epoch, base_lr, args):
    """Decay the learning rate with half-cycle cosine after warmup."""
    warmup_epochs = 100
    min_lr = 0.
    if epoch < warmup_epochs:
        lr = base_lr * epoch / warmup_epochs
    else:
        if args.lr_schedule == "constant":
            lr = base_lr
        elif args.lr_schedule == "cosine":
            lr = min_lr + (base_lr - min_lr) * 0.5 * \
                 (1. + math.cos(math.pi * (epoch - warmup_epochs) / (args.max_epoch - warmup_epochs)))
        else:
            raise NotImplementedError

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
