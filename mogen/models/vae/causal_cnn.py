import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, pad_mode='replicate'):
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation + (1 - stride)
        self.pad_mode = pad_mode
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,  # no padding here
            dilation=dilation
        )

    def forward(self, x):
        if self.pad_mode == 'zero':
            x = nn.functional.pad(x, (self.pad, 0))  # only pad on the left
        else:
            x = nn.functional.pad(x, (self.pad, 0), mode=self.pad_mode)
        return self.conv(x)