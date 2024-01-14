import torch
import torch.nn as nn


class Bias(nn.Module):
    def __init__(self, channels: int):
        super(Bias, self).__init__()
        self.channels = channels
        self.bias = nn.Parameter(torch.zeros(size=(1, channels, 1, 1)))

    def forward(self, x):
        assert x.size()[1] == self.channels, f"unknown size = {x.size()}, expected {self.channels}"
        return x + self.bias
