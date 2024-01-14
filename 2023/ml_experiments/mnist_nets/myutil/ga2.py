import torch.nn as nn


class GA2Mag(nn.Module):
    def forward(self, x):
        channels = x.size()[1]
        assert channels % 4 == 0
        c = channels // 4
        s, x, y, xy = [x[:, c * i: c * (i + 1)] for i in range(4)]
        return (s * s + x * x + y * y + xy * xy) ** 0.5
