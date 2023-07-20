import torch
import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, layer: nn.Module, multiplier: float = 1.0):
        super().__init__()
        self.layer: nn.Module = layer
        self.multiplier: float = multiplier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layer(x) * self.multiplier
