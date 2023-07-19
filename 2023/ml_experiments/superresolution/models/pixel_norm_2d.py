import torch
import torch.nn as nn


class PixelNorm2d(nn.Module):
    """
    Normalizes each pixel independently, so for each pixel x^2 = 1.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (((x ** 2).sum(dim=1, keepdims=True) + 1e-8) / x.size(1)).rsqrt()
