import torch


class DiscriminatorPixelMse:
    def __init__(self, weight: float, inpad_size: int = 0):
        self.mse_loss = torch.nn.MSELoss("mean")
        self.inpad_size: int = inpad_size
        self.weight: float = weight

    def loss(self, y: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        if self.weight == 0.0:
            return torch.zeros(size=[1]).to(y.get_device())

        if self.inpad_size != 0:
            s = self.inpad_size
            y = y[:, :, s:-s, s:-s]
            label = label[:, :, s:-s, s:-s]

        return self.mse_loss(y, label) * self.weight
