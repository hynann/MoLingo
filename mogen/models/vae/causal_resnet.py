import torch.nn as nn
import torch

class nonlinearity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # swish
        return x * torch.sigmoid(x)

class CausalResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, activation='silu', norm=None, dropout=None, pad_mode='reflect'):
        super().__init__()
        self.norm = norm
        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()
        elif activation == "silu":
            # self.activation1 = nonlinearity()
            # self.activation2 = nonlinearity()
            self.activation1 = nn.SiLU()
            self.activation2 = nn.SiLU()
        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()

        self.left_padding = (3 - 1) * dilation
        self.pad_mode = pad_mode

        self.conv1 = nn.Conv1d(n_in, n_state, kernel_size=3, stride=1, padding=0, dilation=dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x_orig = x
        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1)).transpose(-2, -1)
            x = self.activation1(x)
        else:
            x = self.norm1(x)
            x = self.activation1(x)

        if self.pad_mode == 'zero':
            x = nn.functional.pad(x, (self.left_padding, 0)) # only pad on the left
        else:
            x = nn.functional.pad(x, (self.left_padding, 0), mode=self.pad_mode)

        x = self.conv1(x)

        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1)).transpose(-2, -1)
            x = self.activation2(x)
        else:
            x = self.norm2(x)
            x = self.activation2(x)

        x = self.conv2(x)
        x = x + x_orig
        return x

class CausalResnet1D(nn.Module):
    def __init__(self, n_in, n_depth, dilation_growth_rate=1, reverse_dilation=True, activation='relu', norm=None, pad_mode='replicate'):
        super().__init__()

        blocks = [
            CausalResConv1DBlock(
                n_in,
                n_in,
                dilation=dilation_growth_rate ** depth,
                activation=activation,
                norm=norm,
                pad_mode=pad_mode
            ) for depth in range(n_depth)
        ]
        if reverse_dilation:
            blocks = blocks[::-1]

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)