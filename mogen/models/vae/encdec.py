import sys
sys.path.append('.')

import torch.nn as nn
from mogen.models.vae.causal_cnn import CausalConv1d
from mogen.models.vae.causal_resnet import CausalResnet1D

class Encoder(nn.Module):
    def __init__(self,
                 input_emb_width=263, # 263
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 pad_mode='replicate',
                 activation='relu',
                 norm=None):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(CausalConv1d(input_emb_width, width, 3, 1, 1, pad_mode='replicate'))
        blocks.append(nn.SiLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                CausalConv1d(input_dim, width, filter_t, stride_t, 1, pad_mode=pad_mode),
                CausalResnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm, pad_mode=pad_mode),
            )
            blocks.append(block)
        blocks.append(CausalConv1d(width, width, 3, 1, 1, pad_mode=pad_mode))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 pad_mode='replicate',
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []

        blocks.append(CausalConv1d(output_emb_width, width, 3, 1, 1, pad_mode=pad_mode))
        blocks.append(nn.SiLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                CausalResnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm, pad_mode=pad_mode),
                nn.Upsample(scale_factor=2, mode='nearest'),
                CausalConv1d(width, out_dim, 3, 1, 1, pad_mode=pad_mode)
            )
            blocks.append(block)
        blocks.append(CausalConv1d(width, width, 3, 1, 1, pad_mode=pad_mode))
        blocks.append(nn.SiLU())
        blocks.append(CausalConv1d(width, input_emb_width, 3, 1, 1, pad_mode=pad_mode))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.model(x)
        return x.permute(0, 2, 1)