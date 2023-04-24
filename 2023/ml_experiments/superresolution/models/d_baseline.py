from typing import Optional

import torch
import torch.nn as nn

from .random_crop import RandomShiftCrop


class DiscriminatorBaselineNetwork(nn.Module):
    def __init__(self,
                 generator_weight: float,
                 mid_channels: int,
                 inpad_size: int,
                 *,
                 use_batch_norm: bool = True,
                 dropout: Optional[float] = None,
                 deep: bool = False):
        super(DiscriminatorBaselineNetwork, self).__init__()

        self.in_pad = inpad_size
        self.use_batch_norm = use_batch_norm
        self.generator_weight: float = generator_weight
        self.dropout = dropout
        self.random_crop = RandomShiftCrop(4)

        if not deep:
            self.layers = nn.Sequential(
                self.conv(3, mid_channels, kernel_size=5),
                self.conv(mid_channels, mid_channels, kernel_size=3),
                nn.MaxPool2d(2),
                self.conv(mid_channels, mid_channels * 2, kernel_size=3),
                self.conv(mid_channels * 2, mid_channels * 2, kernel_size=3),
                nn.MaxPool2d(2),
                self.conv(mid_channels * 2, mid_channels * 4, kernel_size=3),
                self.conv(mid_channels * 4, mid_channels * 8, kernel_size=1),
                nn.Conv2d(mid_channels * 8, 1, kernel_size=1)
            )
        else:
            self.layers = nn.Sequential(
                self.conv(3, mid_channels, kernel_size=5),
                self.conv(mid_channels, mid_channels, kernel_size=3),
                self.conv(mid_channels, mid_channels, kernel_size=3),
                nn.MaxPool2d(2),
                self.conv(mid_channels, mid_channels * 2, kernel_size=3),
                self.conv(mid_channels * 2, mid_channels * 2, kernel_size=3),
                self.conv(mid_channels * 2, mid_channels * 2, kernel_size=3),
                nn.MaxPool2d(2),
                self.conv(mid_channels * 2, mid_channels * 4, kernel_size=3),
                self.conv(mid_channels * 4, mid_channels * 4, kernel_size=3),
                self.conv(mid_channels * 4, mid_channels * 8, kernel_size=1),
                nn.Conv2d(mid_channels * 8, 1, kernel_size=1)
            )

        self.binary_crossentropy_with_sigmoid = torch.nn.BCEWithLogitsLoss()

    def conv(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> torch.nn.Module:
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding='same', bias=not self.use_batch_norm)]
        if self.use_batch_norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.1))
        if self.dropout is not None:
            layers.append(nn.Dropout(self.dropout))
        return nn.Sequential(*layers)

    def loss_for_discriminator(self,
                               y: torch.Tensor,
                               label: torch.Tensor) -> torch.Tensor:
        y, label = self.random_crop(y, label)

        diff = self(y) - self(label)
        # realness(label) should be >> realness(y)
        zeros = torch.zeros(size=diff.size()).to(diff.device)
        return self.binary_crossentropy_with_sigmoid(diff, zeros)

    def loss(self,
             y: torch.Tensor,
             label: torch.Tensor) -> torch.Tensor:
        if self.generator_weight == 0.0:
            return torch.zeros(size=[1], device=y.get_device())

        y, label = self.random_crop(y, label)

        pred_y = self(y)
        pred_label = self(label)
        y_more_real = pred_y > pred_label

        diff = y_more_real * 9000 + ~y_more_real * (pred_y - pred_label)
        ones = torch.ones(size=diff.size()).to(diff.device)
        return self.binary_crossentropy_with_sigmoid(diff, ones) * self.generator_weight

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        y = self.layers(y)
        return y[:, :, self.in_pad: -self.in_pad, self.in_pad: -self.in_pad]
