from torch import nn as nn

from src.eval_util import eval_model, EvalResult, get_trainable_params


# Simple CNN baseline model
class BaselineCnn(nn.Module):
    def __init__(self, base_channels_count=32):
        super(BaselineCnn, self).__init__()

        activation = nn.LeakyReLU(0.1)

        self.layers = nn.Sequential(
            nn.Conv2d(1, base_channels_count, 5, 1), # 28 -> 24
            activation,
            nn.MaxPool2d(2), # 24 -> 12
            nn.Conv2d(base_channels_count, base_channels_count * 2, 3, 1),  # 12 -> 10
            activation,
            nn.MaxPool2d(2), # 10 -> 5
            nn.Flatten(),
            nn.Linear((base_channels_count * 2) * 5 ** 2, (base_channels_count * 4)),
            activation,
            nn.Linear((base_channels_count * 4), 10)
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == '__main__':
    results = []

    for base_channels_count in [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]:
        print(f"size {base_channels_count}")
        model = BaselineCnn(base_channels_count=base_channels_count).cuda()

        rr = eval_model(model)
        r = EvalResult.most_accurate(rr)

        results.append((r.accuracy, f"* channels = {base_channels_count}, loss = {r.loss:.4f}, accuracy = {r.accuracy:.4f}, params count = {get_trainable_params(model)}"))

    results.sort(reverse=True, key=lambda x: x[0])
    print("final")
    print("\n".join([r for _, r in results]))