import torch
import torch.nn as nn

from .downsampler import DownSampler
from .features_vgg import FeaturesFromVGG16
import random


class DiscriminatorVggMse(nn.Module):
    def __init__(self, weight: float, inpad_size: int, random_shift: int = 8, prescale: int = 1):
        super(DiscriminatorVggMse, self).__init__()

        self.mse_loss = torch.nn.MSELoss("mean")
        self.inpad_size: int = inpad_size
        self.random_shift: int = random_shift
        self.downsampler = DownSampler(prescale) if prescale != 1 else None
        self.vgg = FeaturesFromVGG16(15)
        self.weight: float = weight
        self.requires_grad_(False)

    def crop(self, y: torch.Tensor, initial_size: int) -> torch.Tensor:
        y_size = y.size()[2]
        inpad_size = self.inpad_size * y_size // initial_size
        if inpad_size == 0:
            return y
        return y[:, :, inpad_size:-inpad_size, inpad_size:-inpad_size]

    def loss(self, y: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        if self.weight == 0.0:
            return torch.zeros(size=[1]).to(y.get_device())

        initial_size = y.size()[2]

        rx = random.randrange(0, self.random_shift)
        ry = random.randrange(0, self.random_shift)

        y = y[:, :, ry: initial_size - self.random_shift + ry, rx: initial_size - self.random_shift + rx]
        label = label[:, :, ry: initial_size - self.random_shift + ry, rx: initial_size - self.random_shift + rx]

        if self.downsampler is not None:
            y = torch.clip(y, 0.000001, 1.0)
            label = torch.clip(label, 0.000001, 1.0)
            # downsampler returns Nan for negative input
            y = self.downsampler(y)
            label = self.downsampler(label)

        y_features = self.vgg(y)
        label_features = self.vgg(label)

        result = 0
        for y_f, label_f in zip(y_features, label_features):
            result += self.mse_loss(self.crop(y_f, initial_size), self.crop(label_f, initial_size))

        return result * self.weight
