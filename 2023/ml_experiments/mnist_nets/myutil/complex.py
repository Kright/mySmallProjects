import torch.nn as nn


class ComplexMag(nn.Module):
    def forward(self, x):
        channels = x.size()[1]
        assert channels % 2 == 0
        channels_half = channels // 2
        re, im = x[:, :channels_half], x[:, channels_half:]
        return (re * re + im * im) ** 0.5
