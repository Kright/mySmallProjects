from typing import Tuple

import torch
from torch import nn as nn

from src.eval_util import eval_model, EvalResult, get_trainable_params


# functions for packing and unpacking GA(2) tensors
# GA(2) has 4 components: scalar(0), e1(1), e2(2), e12(12)
# xx = 1, yy = 1, xyxy = -1
def get_0(x: torch.Tensor) -> torch.Tensor:
    return x[:, 0::4]

def get_1(x: torch.Tensor) -> torch.Tensor:
    return x[:, 1::4]

def get_2(x: torch.Tensor) -> torch.Tensor:
    return x[:, 2::4]

def get_12(x: torch.Tensor) -> torch.Tensor:
    return x[:, 3::4]

def get_all(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return get_0(x), get_1(x), get_2(x), get_12(x)

def get_mag(x: torch.Tensor) -> torch.Tensor:
    # Euclidean norm of the multivector
    return torch.sqrt(get_0(x)**2 + get_1(x)**2 + get_2(x)**2 + get_12(x)**2 + 1e-8)

def make_ga2(c0: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor, c12: torch.Tensor) -> torch.Tensor:
    out = torch.stack([c0, c1, c2, c12], dim=2)
    return out.view(c0.shape[0], c0.shape[1] * 4, *c0.shape[2:])


def apply_ga2(fc0, fc1, fc2, fc12, a0, a1, a2, a12):
    # Product components:
    # c0 = a0 b0 + a1 b1 + a2 b2 - a12 b12
    # c1 = a0 b1 + a1 b0 - a2 b12 + a12 b2
    # c2 = a0 b2 + a2 b0 + a1 b12 - a12 b1
    # c12 = a0 b12 + a12 b0 + a1 b2 - a2 b1
    out0 = fc0(a0) + fc1(a1) + fc2(a2) - fc12(a12)
    out1 = fc1(a0) + fc0(a1) + fc2(a12) - fc12(a2)
    out2 = fc2(a0) + fc0(a2) + fc12(a1) - fc1(a12)
    out12 = fc12(a0) + fc0(a12) + fc2(a1) - fc1(a2)
    return out0, out1, out2, out12


class ToGa2(nn.Module):
    def forward(self, x):
        zeros = torch.zeros_like(x)
        return make_ga2(x, zeros, zeros, zeros)


class Ga2Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv12 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        # x: [Batch, 4*in_channels, Height, Width]
        a0, a1, a2, a12 = get_all(x)
        out0, out1, out2, out12 = apply_ga2(self.conv0, self.conv1, self.conv2, self.conv12, a0, a1, a2, a12)
        return make_ga2(out0, out1, out2, out12)


class Ga2Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc0 = nn.Linear(in_features, out_features)
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(in_features, out_features)
        self.fc12 = nn.Linear(in_features, out_features)

    def forward(self, x):
        # x: [Batch, 4*in_features]
        a0, a1, a2, a12 = get_all(x)
        out0, out1, out2, out12 = apply_ga2(self.fc0, self.fc1, self.fc2, self.fc12, a0, a1, a2, a12)
        return make_ga2(out0, out1, out2, out12)


class ModReLU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.b = nn.Parameter(torch.randn(channels) * 0.01)

    def forward(self, x):
        a0, a1, a2, a12 = get_all(x)
        mag = torch.sqrt(a0**2 + a1**2 + a2**2 + a12**2 + 1e-8)
        
        C = a0.shape[1]
        b_shape = [1, C] + [1] * (mag.ndim - 2)
        b = self.b.view(*b_shape)
        
        norm_mag = nn.functional.relu(mag + b)
        scale = norm_mag / (mag + 1e-8)
        
        return make_ga2(a0 * scale, a1 * scale, a2 * scale, a12 * scale)


class SoftModReLU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        a0, a1, a2, a12 = get_all(x)
        mag = torch.sqrt(a0**2 + a1**2 + a2**2 + a12**2 + 1e-8)

        C = a0.shape[1]
        b_shape = [1, C] + [1] * (mag.ndim - 2)
        b = self.b.view(*b_shape)

        soft_mag = nn.functional.softplus(mag + b)
        scale = soft_mag / (mag + 1e-8)
        return make_ga2(a0 * scale, a1 * scale, a2 * scale, a12 * scale)


class LogModReLU(nn.Module):
    def __init__(self, channels=None):
        super().__init__()

    def forward(self, x):
        a0, a1, a2, a12 = get_all(x)
        mag = torch.sqrt(a0**2 + a1**2 + a2**2 + a12**2 + 1e-8)
        scale = (mag + 1.0).log() / (mag + 1e-8)
        return make_ga2(a0 * scale, a1 * scale, a2 * scale, a12 * scale)


class Ga2Sigmoid(nn.Module):
    def forward(self, x):
        a0, a1, a2, a12 = get_all(x)
        mag = torch.sqrt(a0**2 + a1**2 + a2**2 + a12**2 + 1e-8)
        scale = 0.5 / (1.0 + mag)
        return make_ga2(a0 * scale + 0.5, a1 * scale, a2 * scale, a12 * scale)


class Ga2SoftRooting(nn.Module):
    def forward(self, x):
        a0, a1, a2, a12 = get_all(x)
        mag_sq = a0**2 + a1**2 + a2**2 + a12**2
        scale = 1.0 / torch.sqrt(1.0 + mag_sq)
        return make_ga2(a0 * scale, a1 * scale, a2 * scale, a12 * scale)


class MagnitudeBasedGLU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        a0, a1, a2, a12 = get_all(x)
        mag = torch.sqrt(a0**2 + a1**2 + a2**2 + a12**2 + 1e-8)

        C = a0.shape[1]
        b_shape = [1, C] + [1] * (mag.ndim - 2)
        b = self.bias.view(*b_shape)

        multiplier = torch.sigmoid(mag + b)
        return make_ga2(a0 * multiplier, a1 * multiplier, a2 * multiplier, a12 * multiplier)


class Ga2ModGLU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.make_gates = Ga2Conv2d(channels, channels, 1)

    def forward(self, x):
        if x.ndim == 2:
            gates = self.make_gates(x.view(*x.shape, 1, 1)).view(x.shape[0], -1)
        else:
            gates = self.make_gates(x)

        g0, g1, g2, g12 = get_all(gates)
        # Analogy to ComplexModGLU: first component sigmoid, others tanh
        mult = torch.sigmoid(g0) * torch.tanh(g1) * torch.tanh(g2) * torch.tanh(g12)

        a0, a1, a2, a12 = get_all(x)
        return make_ga2(a0 * mult, a1 * mult, a2 * mult, a12 * mult)


class Cardioid(nn.Module):
    def forward(self, x):
        a0, a1, a2, a12 = get_all(x)
        mag = torch.sqrt(a0**2 + a1**2 + a2**2 + a12**2 + 1e-8)
        scale = 0.5 * (1 + a0 / mag)
        return make_ga2(a0 * scale, a1 * scale, a2 * scale, a12 * scale)


class zReLU(nn.Module):
    def forward(self, x):
        a0, a1, a2, a12 = get_all(x)
        mask = (a0 > 0) & (a1 > 0) & (a2 > 0) & (a12 > 0)
        mask = mask.to(x.dtype)
        return make_ga2(a0 * mask, a1 * mask, a2 * mask, a12 * mask)


class SplitELU(nn.Module):
    def forward(self, x):
        a0, a1, a2, a12 = get_all(x)
        return make_ga2(nn.functional.elu(a0), nn.functional.elu(a1), nn.functional.elu(a2), nn.functional.elu(a12))


class Ga2MaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size, stride, padding, return_indices=True)

    def forward(self, x):
        a0, a1, a2, a12 = get_all(x)
        mag_sq = a0**2 + a1**2 + a2**2 + a12**2
        _, indices = self.maxpool(mag_sq)

        B, C, out_h, out_w = indices.shape
        
        a0_f = a0.view(B, C, -1)
        a1_f = a1.view(B, C, -1)
        a2_f = a2.view(B, C, -1)
        a12_f = a12.view(B, C, -1)
        indices_flat = indices.view(B, C, -1)
        
        o0 = torch.gather(a0_f, 2, indices_flat).view(B, C, out_h, out_w)
        o1 = torch.gather(a1_f, 2, indices_flat).view(B, C, out_h, out_w)
        o2 = torch.gather(a2_f, 2, indices_flat).view(B, C, out_h, out_w)
        o12 = torch.gather(a12_f, 2, indices_flat).view(B, C, out_h, out_w)
        
        return make_ga2(o0, o1, o2, o12)


class ToReal(nn.Module):
    def forward(self, x):
        return get_0(x)


class Ga2Cnn(nn.Module):
    def __init__(self, get_activation=None, base_channels_count=32):
        super(Ga2Cnn, self).__init__()

        if get_activation is None:
            def get_activation(channels):
                return nn.LeakyReLU(0.1)

        self.layers = nn.Sequential(
            ToGa2(),
            Ga2Conv2d(1, base_channels_count, 5, 1),
            get_activation(base_channels_count),
            Ga2MaxPool2d(2), 
            Ga2Conv2d(base_channels_count, base_channels_count * 2, 3, 1),
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
        "Ga2Sigmoid": lambda channels: Ga2Sigmoid(),
        "Ga2SoftRooting": lambda channels: Ga2SoftRooting(),
        "Ga2ModGLU": lambda channels: Ga2ModGLU(channels),
        "MagnitudeBasedGLU": lambda channels: MagnitudeBasedGLU(channels),
        "Cardioid": lambda channels: Cardioid(),
        "zReLU": lambda channels: zReLU(),
        "PReLU": lambda channels: nn.PReLU(channels * 4),
        "SplitTanh": lambda channels: nn.Tanh(),
        "SplitELU": lambda channels: SplitELU(),
        "nn.ReLU": lambda channels: nn.ReLU(),
        "nn.LeakyReLU(0.1)": lambda channels: nn.LeakyReLU(0.1),
        "ModReLU": lambda channels: ModReLU(channels),
        "SoftModReLU": lambda channels: SoftModReLU(channels),
        "LogModReLU": lambda channels: LogModReLU(channels),
    }

    results = []

    for name, get_act in activations.items():
        print(f"activation {name}")
        model = Ga2Cnn(get_activation=get_act, base_channels_count=16).cuda()

        rr = eval_model(model)
        r = EvalResult.most_accurate(rr)

        results.append((r.accuracy, f"activation {name}, loss = {r.loss:.4f}, accuracy = {r.accuracy:.4f}, params count = {get_trainable_params(model)}"))

    results.sort(reverse=True, key=lambda x: x[0])
    print("final")
    print("\n".join([r for _, r in results]))

