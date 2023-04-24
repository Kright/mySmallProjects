import torch.nn as nn
import torch

# inspired by code from here https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/esrgan/models.py
# and here https://github.com/Lornatang/ESRGAN-PyTorch/blob/main/model.py


class DenseResidualBlock(nn.Module):
    def __init__(self, clannels, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        blocks = []
        for i in range(1, 5):
            blocks.append(nn.Sequential(
                nn.Conv2d(i * clannels, clannels, kernel_size=3, padding='same', bias=True),
                nn.LeakyReLU(0.1)
            ))

        self.blocks = torch.nn.ParameterList(blocks)
        self.last_conv = nn.Conv2d(5 * clannels, clannels, kernel_size=3, padding='same', bias=True)

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)

        return x + self.last_conv(inputs) * self.res_scale


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters),
            DenseResidualBlock(filters),
            DenseResidualBlock(filters)
        )

    def forward(self, x):
        return x + self.dense_blocks(x) * self.res_scale


class GeneratorRRDB(nn.Module):
    def __init__(self, channels: int = 64, num_res_blocks=16, upscale: int = 4):
        super(GeneratorRRDB, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3, channels, kernel_size=5, padding='same'))
        for i in range(num_res_blocks):
            layers.append(ResidualInResidualDenseBlock(channels))

        layers.append(nn.Conv2d(channels, channels * 2, kernel_size=3, padding='same', bias=True))
        layers.append(nn.LeakyReLU(0.1))

        self.layers = nn.Sequential(*layers)

        # this layer lr should be 0.1 times less
        self.pixel_shuffle = nn.Sequential(
            nn.Conv2d(channels * 2, 3 * upscale ** 2, kernel_size=3, padding='same'),
            nn.PixelShuffle(upscale_factor=4),
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.pixel_shuffle(x)
        return x