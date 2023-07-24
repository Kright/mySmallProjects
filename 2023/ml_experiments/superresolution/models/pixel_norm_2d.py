import torch
import torch.nn as nn


class PixelNorm2d(nn.Module):
    """
    Normalizes each pixel independently, so for each pixel x^2 = 1.
    Same as RMSNorm, but for each pixel independently
    supports torch.jit.script
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (x.pow(2).mean(dim=1, keepdim=True) + 1e-8).rsqrt()
