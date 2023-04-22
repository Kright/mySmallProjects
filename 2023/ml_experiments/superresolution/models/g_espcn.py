import torch
import torch.nn as nn


class GeneratorESPCN(nn.Module):
    def __init__(self, channels: int, upscale: int):
        super(GeneratorESPCN, self).__init__()

        # original network contains 3 layers with tanh activation, but I use leakyReLU
        self.layers = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=5, padding='same'),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channels, channels, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.1),
        )

        # this layer lr should be 0.1 times less
        self.pixel_shuffle = nn.Sequential(
            nn.Conv2d(channels, 3 * upscale ** 2, kernel_size=3, padding='same'),
            nn.PixelShuffle(upscale),
        )
        # no activation after last layer

    def forward(self, x: torch.Tensor):
        x = self.layers(x)
        x = self.pixel_shuffle(x)
        return x