from typing import Tuple

import torch
from torch import nn as nn

from src.eval_util import eval_model, EvalResult, get_trainable_params
from src.ga2 import ToGa2, Ga2Conv2d, Ga2MaxPool2d, Ga2Linear, ToReal, Ga2Sigmoid, Ga2SoftRooting, Ga2ModGLU, Cardioid, \
    zReLU, MagnitudeBasedGLU, SplitELU, ModReLU, SoftModReLU, LogModReLU


class Ga2CnnBN(nn.Module):
    def __init__(self, get_activation=None, base_channels_count=32, use_bn = True):
        super(Ga2CnnBN, self).__init__()

        if get_activation is None:
            def get_activation(channels):
                return nn.LeakyReLU(0.1)

        def make_bn(channels):
            return nn.BatchNorm2d(channels * 4) if use_bn else nn.Identity()

        self.layers = nn.Sequential(
            ToGa2(),
            Ga2Conv2d(1, base_channels_count, 5, 1),
            make_bn(base_channels_count),
            get_activation(base_channels_count),
            Ga2MaxPool2d(2),
            Ga2Conv2d(base_channels_count, base_channels_count * 2, 3, 1),
            make_bn(base_channels_count * 2),
            get_activation(base_channels_count * 2),
            Ga2MaxPool2d(2),
            nn.Flatten(),
            Ga2Linear((base_channels_count * 2) * 5 ** 2, base_channels_count * 4),
            get_activation(base_channels_count * 4),
            Ga2Linear(base_channels_count * 4, 10),
            ToReal()
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == '__main__':
    activations = {
        "Ga2ModGLU": lambda channels: Ga2ModGLU(channels),
        "Cardioid": lambda channels: Cardioid(),
        "PReLU": lambda channels: nn.PReLU(channels * 4),
        "nn.LeakyReLU(0.1)": lambda channels: nn.LeakyReLU(0.1),
    }

    results = []

    for use_bn in [True, False]:
        for name, get_act in activations.items():
            print(f"activation {name}")
            model = Ga2CnnBN(get_activation=get_act, base_channels_count=8, use_bn=use_bn).cuda()

            rr = eval_model(model, lr = 0.01, epochs=10)
            r = EvalResult.most_accurate(rr)

            results.append((r.accuracy, f"activation {name}, use_bn = {use_bn}, loss = {r.loss:.4f}, accuracy = {r.accuracy:.4f}, params count = {get_trainable_params(model)}"))

    results.sort(reverse=True, key=lambda x: x[0])
    print("final")
    print("\n".join([r for _, r in results]))


# activation Cardioid, use_bn = True, loss = 0.0350, accuracy = 0.9924, params count = 58344
# activation nn.LeakyReLU(0.1), use_bn = True, loss = 0.0336, accuracy = 0.9912, params count = 58344
# activation PReLU, use_bn = True, loss = 0.0315, accuracy = 0.9910, params count = 58568
# activation Ga2ModGLU, use_bn = True, loss = 0.0341, accuracy = 0.9904, params count = 63944
# activation nn.LeakyReLU(0.1), use_bn = False, loss = 0.0363, accuracy = 0.9897, params count = 58152
# activation Cardioid, use_bn = False, loss = 0.0356, accuracy = 0.9894, params count = 58152
# activation Ga2ModGLU, use_bn = False, loss = 0.0339, accuracy = 0.9889, params count = 63752
# activation PReLU, use_bn = False, loss = 0.0378, accuracy = 0.9887, params count = 58376
