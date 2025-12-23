from typing import Tuple

import torch
from torch import nn as nn

from src.eval_util import eval_model, EvalResult, get_trainable_params


# functions for packing and unpacking GA(3) tensors
# GA(3) has 8 components: scalar(0), e1(1), e2(2), e3(3), e12(12), e23(23), e13(13), e123(123)
# e1^2 = e2^2 = e3^2 = 1

def get_0(x: torch.Tensor) -> torch.Tensor:
    return x[:, 0::8]

def get_1(x: torch.Tensor) -> torch.Tensor:
    return x[:, 1::8]

def get_2(x: torch.Tensor) -> torch.Tensor:
    return x[:, 2::8]

def get_3(x: torch.Tensor) -> torch.Tensor:
    return x[:, 3::8]

def get_12(x: torch.Tensor) -> torch.Tensor:
    return x[:, 4::8]

def get_23(x: torch.Tensor) -> torch.Tensor:
    return x[:, 5::8]

def get_13(x: torch.Tensor) -> torch.Tensor:
    return x[:, 6::8]

def get_123(x: torch.Tensor) -> torch.Tensor:
    return x[:, 7::8]

def get_all(x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    return get_0(x), get_1(x), get_2(x), get_3(x), get_12(x), get_23(x), get_13(x), get_123(x)

def get_mag(x: torch.Tensor) -> torch.Tensor:
    # Euclidean norm of the multivector
    return torch.sqrt(sum(c**2 for c in get_all(x)) + 1e-8)

def make_ga3(c0, c1, c2, c3, c12, c23, c13, c123) -> torch.Tensor:
    out = torch.stack([c0, c1, c2, c3, c12, c23, c13, c123], dim=2)
    return out.view(c0.shape[0], c0.shape[1] * 8, *c0.shape[2:])


def apply_ga3(fc0, fc1, fc2, fc3, fc12, fc23, fc13, fc123, a0, a1, a2, a3, a12, a23, a13, a123):
    # Product components:
    # c0 = a0 b0 + a1 b1 + a2 b2 + a3 b3 - a12 b12 - a23 b23 - a13 b13 - a123 b123
    out0 = fc0(a0) + fc1(a1) + fc2(a2) + fc3(a3) - fc12(a12) - fc23(a23) - fc13(a13) - fc123(a123)
    
    # c1 = a0 b1 + a1 b0 + a12 b2 - a2 b12 + a13 b3 - a3 b13 - a123 b23 - a23 b123
    out1 = fc1(a0) + fc0(a1) + fc2(a12) - fc12(a2) + fc3(a13) - fc13(a3) - fc23(a123) - fc123(a23)
    
    # c2 = a0 b2 + a2 b0 - a12 b1 + a1 b12 + a23 b3 - a3 b23 + a123 b13 + a13 b123
    out2 = fc2(a0) + fc0(a2) - fc1(a12) + fc12(a1) + fc3(a23) - fc23(a3) + fc13(a123) + fc123(a13)
    
    # c3 = a0 b3 + a3 b0 - a13 b1 + a1 b13 - a23 b2 + a2 b23 - a123 b12 - a12 b123
    out3 = fc3(a0) + fc0(a3) - fc1(a13) + fc13(a1) - fc2(a23) + fc23(a2) - fc12(a123) - fc123(a12)
    
    # c12 = a0 b12 + a12 b0 + a1 b2 - a2 b1 + a123 b3 + a3 b123 + a23 b13 - a13 b23
    out12 = fc12(a0) + fc0(a12) + fc2(a1) - fc1(a2) + fc3(a123) + fc123(a3) + fc13(a23) - fc23(a13)
    
    # c23 = a0 b23 + a23 b0 + a2 b3 - a3 b2 + a123 b1 + a1 b123 + a13 b12 - a12 b13
    out23 = fc23(a0) + fc0(a23) + fc3(a2) - fc2(a3) + fc1(a123) + fc123(a1) + fc12(a13) - fc13(a12)
    
    # c13 = a0 b13 + a13 b0 + a1 b3 - a3 b1 - a123 b2 - a2 b123 + a12 b23 - a23 b12
    out13 = fc13(a0) + fc0(a13) + fc3(a1) - fc1(a3) - fc2(a123) - fc123(a2) + fc23(a12) - fc12(a23)
    
    # c123 = a0 b123 + a123 b0 + a1 b23 + a23 b1 - a2 b13 - a13 b2 + a3 b12 + a12 b3
    out123 = fc123(a0) + fc0(a123) + fc23(a1) + fc1(a23) - fc13(a2) - fc2(a13) + fc12(a3) + fc3(a12)
    
    return out0, out1, out2, out3, out12, out23, out13, out123


class ToGa3(nn.Module):
    def forward(self, x):
        zeros = torch.zeros_like(x)
        return make_ga3(x, zeros, zeros, zeros, zeros, zeros, zeros, zeros)


class Ga3Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv12 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv23 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv13 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv123 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        a0, a1, a2, a3, a12, a23, a13, a123 = get_all(x)
        out = apply_ga3(self.conv0, self.conv1, self.conv2, self.conv3, self.conv12, self.conv23, self.conv13, self.conv123,
                        a0, a1, a2, a3, a12, a23, a13, a123)
        return make_ga3(*out)


class Ga3Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc0 = nn.Linear(in_features, out_features)
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(in_features, out_features)
        self.fc3 = nn.Linear(in_features, out_features)
        self.fc12 = nn.Linear(in_features, out_features)
        self.fc23 = nn.Linear(in_features, out_features)
        self.fc13 = nn.Linear(in_features, out_features)
        self.fc123 = nn.Linear(in_features, out_features)

    def forward(self, x):
        a0, a1, a2, a3, a12, a23, a13, a123 = get_all(x)
        out = apply_ga3(self.fc0, self.fc1, self.fc2, self.fc3, self.fc12, self.fc23, self.fc13, self.fc123,
                        a0, a1, a2, a3, a12, a23, a13, a123)
        return make_ga3(*out)


class ModReLU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.b = nn.Parameter(torch.randn(channels) * 0.01)

    def forward(self, x):
        components = get_all(x)
        mag = torch.sqrt(sum(c**2 for c in components) + 1e-8)
        
        C = components[0].shape[1]
        b_shape = [1, C] + [1] * (mag.ndim - 2)
        b = self.b.view(*b_shape)
        
        norm_mag = nn.functional.relu(mag + b)
        scale = norm_mag / (mag + 1e-8)
        
        return make_ga3(*(c * scale for c in components))


class SoftModReLU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        components = get_all(x)
        mag = torch.sqrt(sum(c**2 for c in components) + 1e-8)

        C = components[0].shape[1]
        b_shape = [1, C] + [1] * (mag.ndim - 2)
        b = self.b.view(*b_shape)

        soft_mag = nn.functional.softplus(mag + b)
        scale = soft_mag / (mag + 1e-8)
        return make_ga3(*(c * scale for c in components))


class LogModReLU(nn.Module):
    def __init__(self, channels=None):
        super().__init__()

    def forward(self, x):
        components = get_all(x)
        mag = torch.sqrt(sum(c**2 for c in components) + 1e-8)
        scale = (mag + 1.0).log() / (mag + 1e-8)
        return make_ga3(*(c * scale for c in components))


class Ga3Sigmoid(nn.Module):
    def forward(self, x):
        components = get_all(x)
        mag = torch.sqrt(sum(c**2 for c in components) + 1e-8)
        scale = 0.5 / (1.0 + mag)
        return make_ga3(components[0] * scale + 0.5, *(c * scale for c in components[1:]))


class Ga3SoftRooting(nn.Module):
    def forward(self, x):
        components = get_all(x)
        mag_sq = sum(c**2 for c in components)
        scale = 1.0 / torch.sqrt(1.0 + mag_sq)
        return make_ga3(*(c * scale for c in components))


class MagnitudeBasedGLU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        components = get_all(x)
        mag = torch.sqrt(sum(c**2 for c in components) + 1e-8)

        C = components[0].shape[1]
        b_shape = [1, C] + [1] * (mag.ndim - 2)
        b = self.bias.view(*b_shape)

        multiplier = torch.sigmoid(mag + b)
        return make_ga3(*(c * multiplier for c in components))


class Ga3ModGLU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.make_gates = Ga3Conv2d(channels, channels, 1)

    def forward(self, x):
        if x.ndim == 2:
            gates = self.make_gates(x.view(*x.shape, 1, 1)).view(x.shape[0], -1)
        else:
            gates = self.make_gates(x)

        gs = get_all(gates)
        # First component sigmoid, others tanh
        mult = torch.sigmoid(gs[0])
        for g in gs[1:]:
            mult = mult * torch.tanh(g)

        components = get_all(x)
        return make_ga3(*(c * mult for c in components))


class Cardioid(nn.Module):
    def forward(self, x):
        components = get_all(x)
        mag = torch.sqrt(sum(c**2 for c in components) + 1e-8)
        scale = 0.5 * (1 + components[0] / mag)
        return make_ga3(*(c * scale for c in components))


class zReLU(nn.Module):
    def forward(self, x):
        components = get_all(x)
        mask = components[0] > 0
        for c in components[1:]:
            mask = mask & (c > 0)
        mask = mask.to(x.dtype)
        return make_ga3(*(c * mask for c in components))


class SplitELU(nn.Module):
    def forward(self, x):
        components = get_all(x)
        return make_ga3(*(nn.functional.elu(c) for c in components))


class Ga3MaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size, stride, padding, return_indices=True)

    def forward(self, x):
        components = get_all(x)
        mag_sq = sum(c**2 for c in components)
        _, indices = self.maxpool(mag_sq)

        B, C, out_h, out_w = indices.shape
        indices_flat = indices.view(B, C, -1)
        
        outputs = []
        for c in components:
            c_f = c.view(B, C, -1)
            o = torch.gather(c_f, 2, indices_flat).view(B, C, out_h, out_w)
            outputs.append(o)
        
        return make_ga3(*outputs)


class ToReal(nn.Module):
    def forward(self, x):
        return get_0(x)


class Ga3Cnn(nn.Module):
    def __init__(self, get_activation=None, base_channels_count=32):
        super(Ga3Cnn, self).__init__()

        if get_activation is None:
            def get_activation(channels):
                return nn.LeakyReLU(0.1)

        self.layers = nn.Sequential(
            ToGa3(),
            Ga3Conv2d(1, base_channels_count, 5, 1),
            get_activation(base_channels_count),
            Ga3MaxPool2d(2), 
            Ga3Conv2d(base_channels_count, base_channels_count * 2, 3, 1),
            get_activation(base_channels_count * 2),
            Ga3MaxPool2d(2),
            nn.Flatten(),
            Ga3Linear((base_channels_count * 2) * 5 ** 2, base_channels_count * 4),
            get_activation(base_channels_count * 4),
            Ga3Linear(base_channels_count * 4, 10),
            ToReal()
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == '__main__':
    activations = {
        "Ga3Sigmoid": lambda channels: Ga3Sigmoid(),
        "Ga3SoftRooting": lambda channels: Ga3SoftRooting(),
        "Ga3ModGLU": lambda channels: Ga3ModGLU(channels),
        "MagnitudeBasedGLU": lambda channels: MagnitudeBasedGLU(channels),
        "Cardioid": lambda channels: Cardioid(),
        "zReLU": lambda channels: zReLU(),
        "PReLU": lambda channels: nn.PReLU(channels * 8),
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
        model = Ga3Cnn(get_activation=get_act, base_channels_count=2).cuda()

        rr = eval_model(model)
        r = EvalResult.most_accurate(rr)

        results.append((r.accuracy, f"activation {name}, loss = {r.loss:.4f}, accuracy = {r.accuracy:.4f}, params count = {get_trainable_params(model)}"))

    results.sort(reverse=True, key=lambda x: x[0])
    print("final")
    print("\n".join([r for _, r in results]))
