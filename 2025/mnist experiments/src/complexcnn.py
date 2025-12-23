from typing import Tuple

import torch
from torch import nn as nn

from src.eval_util import eval_model, EvalResult, get_trainable_params


# functions for packing and unpacking complex tensors
def get_re(x: torch.Tensor) -> torch.Tensor:
    return x[:, 0::2]

def get_im(x: torch.Tensor) -> torch.Tensor:
    return x[:, 1::2]

def get_re_im(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return get_re(x), get_im(x)

def get_mag(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(get_re(x)**2 + get_im(x)**2)

def make_complex(re: torch.Tensor, im: torch.Tensor) -> torch.Tensor:
    out = torch.stack([re, im], dim=2)
    return out.view(re.shape[0], re.shape[1] * 2, *re.shape[2:])



class ToComplex(nn.Module):
    def forward(self, x):
        return make_complex(x, torch.zeros_like(x))


class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        # x: [Batch, 2*in_channels, Height, Width]
        x_r = get_re(x)
        x_i = get_im(x)
        out_r = self.conv_r(x_r) - self.conv_i(x_i)
        out_i = self.conv_r(x_i) + self.conv_i(x_r)
        
        return make_complex(out_r, out_i)


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc_r = nn.Linear(in_features, out_features)
        self.fc_i = nn.Linear(in_features, out_features)

    def forward(self, x):
        # x: [Batch, 2*in_features]
        x_r = get_re(x)
        x_i = get_im(x)
        out_r = self.fc_r(x_r) - self.fc_i(x_i)
        out_i = self.fc_r(x_i) + self.fc_i(x_r)
        
        return make_complex(out_r, out_i)


class ModReLU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.b = nn.Parameter(torch.randn(channels) * 0.01)

    def forward(self, x):
        # x is interleaved: [B, 2*C, ...]
        x_r, x_i = get_re(x), get_im(x)
        mag = torch.sqrt(x_r**2 + x_i**2 + 1e-8)
        
        C = x_r.shape[1]
        b_shape = [1, C] + [1] * (mag.ndim - 2)
        b = self.b.view(*b_shape)
        
        norm_mag = nn.functional.relu(mag + b)
        scale = norm_mag / (mag + 1e-8)
        
        return make_complex(x_r * scale, x_i * scale)


class SoftModReLU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x_r, x_i = get_re(x), get_im(x)
        mag = torch.sqrt(x_r**2 + x_i**2 + 1e-8)

        C = x_r.shape[1]
        b_shape = [1, C] + [1] * (mag.ndim - 2)
        b = self.b.view(*b_shape)

        # Вместо ReLU(mag + b) используем Softplus
        soft_mag = nn.functional.softplus(mag + b)
        scale = soft_mag / (mag + 1e-8)
        return make_complex(x_r * scale, x_i * scale)


class LogModReLU(nn.Module):
    def __init__(self, channels):
        super().__init__()

    def forward(self, x):
        # x is interleaved: [B, 2*C, ...]
        x_r, x_i = get_re(x), get_im(x)
        mag = torch.sqrt(x_r ** 2 + x_i ** 2 + 1e-8)
        scale = (mag + 1.0).log() / (mag + 1e-8)
        return make_complex(x_r * scale, x_i * scale)


class Cardioid(nn.Module):
    def forward(self, x):
        # x: [B, 2*C, ...] interleaved
        x_r, x_i = get_re(x), get_im(x)
        phi = torch.atan2(x_i, x_r)
        scale = 0.5 * (1 + torch.cos(phi))
        return make_complex(x_r * scale, x_i * scale)


class zReLU(nn.Module):
    def forward(self, x):
        # x: [B, 2*C, ...] interleaved
        x_r, x_i = get_re(x), get_im(x)
        mask = (x_r > 0) & (x_i > 0)
        mask = mask.to(x.dtype)
        return make_complex(x_r * mask, x_i * mask)


class MagnitudeBasedGLU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x_r, x_i = get_re_im(x)
        mag = get_mag(x)

        C = x_r.shape[1]
        b_shape = [1, C] + [1] * (mag.ndim - 2)
        b = self.bias.view(*b_shape)

        multiplier = nn.functional.sigmoid(mag + b)
        return make_complex(x_r * multiplier, x_i * multiplier)


class ComplexModGLU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        assert channels % 2 == 0, "channels must be even"
        self.make_gates = ComplexConv2d(channels, channels, 1)

    def forward(self, x):
        """preserves channels count, but computers gates based on the same x"""
        if x.ndim == 2:
            # handle [Batch, 2*channels]
            gates = self.make_gates(x.view(*x.shape, 1, 1)).view(x.shape[0], -1)
        else:
            # handle [Batch, 2*channels, H, W]
            gates = self.make_gates(x)

        mult = nn.functional.sigmoid(get_re(gates)) * nn.functional.tanh(get_im(gates))

        x_r, x_i = get_re_im(x)
        return make_complex(x_r * mult, x_i * mult)


class SplitELU(nn.Module):
    def forward(self, x):
        x_r, x_i = get_re_im(x)
        return make_complex(nn.functional.elu(x_r), nn.functional.elu(x_i))


class ComplexMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size, stride, padding, return_indices=True)

    def forward(self, x):
        # x: [B, 2*C, H, W] interleaved
        x_r, x_i = get_re_im(x)
        mag = x_r**2 + x_i**2
        _, indices = self.maxpool(mag)

        B, C, out_h, out_w = indices.shape
        
        # Flatten and gather
        r_flat = x_r.view(B, C, -1)
        i_flat = x_i.view(B, C, -1)
        indices_flat = indices.view(B, C, -1)
        
        out_r = torch.gather(r_flat, 2, indices_flat).view(B, C, out_h, out_w)
        out_i = torch.gather(i_flat, 2, indices_flat).view(B, C, out_h, out_w)
        
        return make_complex(out_r, out_i)


class ComplexSigmoid(nn.Module):
    """
    Bounded complex sigmoid: f(z) = 0.5 * (z / (1 + |z|)) + 0.5
    Keeps the phase but squashes the magnitude into [0, 1] range.
    """
    def forward(self, x):
        x_r, x_i = get_re_im(x)
        mag = torch.sqrt(x_r**2 + x_i**2 + 1e-8)
        scale = 0.5 / (1.0 + mag)
        # Сдвиг 0.5 применяется к вещественной части для центрирования
        return make_complex(x_r * scale + 0.5, x_i * scale)


class ComplexSoftRooting(nn.Module):
    """
    Holomorphic-ish activation: f(z) = z / sqrt(1 + |z|^2)
    It's a smooth version of complex tanh that doesn't explode.
    """

    def forward(self, x):
        x_r, x_i = get_re_im(x)
        # |z|^2
        mag_sq = x_r ** 2 + x_i ** 2
        scale = 1.0 / torch.sqrt(1.0 + mag_sq)
        return make_complex(x_r * scale, x_i * scale)


class ToReal(nn.Module):
    def forward(self, x):
        return get_re(x)


# Simple CNN baseline model
class ComplexCnn(nn.Module):
    def __init__(self, get_activation=None, base_channels_count=32):
        super(ComplexCnn, self).__init__()

        if get_activation is None:
            # nn.LeakyReLU (ComplexLeakyReLU) is used to match BaselineCnn
            # Other options: ModReLU(channels), Cardioid(), zReLU()
            def get_activation(channels):
                return nn.LeakyReLU(0.1)

        self.layers = nn.Sequential(
            ToComplex(),
            ComplexConv2d(1, base_channels_count, 5, 1), # 28 -> 24
            get_activation(base_channels_count),
            ComplexMaxPool2d(2), # 24 -> 12
            ComplexConv2d(base_channels_count, base_channels_count * 2, 3, 1),  # 12 -> 10
            get_activation(base_channels_count * 2),
            ComplexMaxPool2d(2), # 10 -> 5
            nn.Flatten(),
            ComplexLinear((base_channels_count * 2) * 5 ** 2, (base_channels_count * 4)),
            get_activation((base_channels_count * 4)),
            ComplexLinear((base_channels_count * 4), 10),
            ToReal()
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == '__main__':
    activations = {
        "ComplexSigmoid": lambda channels: ComplexSigmoid(),
        "ComplexSoftRooting": lambda channels: ComplexSoftRooting(),
        "ComplexModGLU": lambda channels: ComplexModGLU(channels),
        "MagnitudeBasedGLU": lambda channels: MagnitudeBasedGLU(channels),
        "PreLU": lambda channels: nn.PReLU(channels * 2),
        "SplitTanh": lambda channels: nn.Tanh(),
        "SplitELU": lambda channels: SplitELU(),
        "nn.ReLU": lambda channels: nn.ReLU(),
        "nn.LeakyReLU(0.1)": lambda channels: nn.LeakyReLU(0.1),
        "ModReLU": lambda channels: ModReLU(channels),
        "SoftModReLU": lambda channels: SoftModReLU(channels),
        "LogModReLU": lambda channels: LogModReLU(channels),
        "Cardioid": lambda channels: Cardioid(),
        "zReLU": lambda channels: zReLU(),
    }

    results = []

    for name, get_act in activations.items():
        print(f"activation {name}")
        model = ComplexCnn(get_activation=get_act, base_channels_count=32).cuda()

        # print_model_summary(model)
        rr = eval_model(model)
        r = EvalResult.most_accurate(rr)

        results.append((r.accuracy, f"activation {name}, loss = {r.loss:.4f}, accuracy = {r.accuracy:.4f}, params count = {get_trainable_params(model)}"))

    results.sort(reverse=True, key=lambda x: x[0])
    print("final")
    print("\n".join([r for _, r in results]))
