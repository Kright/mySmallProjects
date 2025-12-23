from typing import List

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from dataclasses import dataclass


@dataclass
class EvalResult:
    loss: float
    accuracy: float

    @staticmethod
    def most_accurate(lst: List['EvalResult']) -> 'EvalResult':
        return max(lst, key=lambda x: x.accuracy)


def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        # if batch_idx % 100 == 0:
        #     print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
        #           f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, test_loader) -> EvalResult:
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.2f}%)')

    return EvalResult(test_loss, correct / len(test_loader.dataset))


def get_trainable_params(model_or_layer):
    return sum(p.numel() for p in model_or_layer.parameters() if p.requires_grad)


def print_model_summary(model):
    print(f"Network total_params = {get_trainable_params(model)}, layers:")
    for i, layer in enumerate(model.layers):
        shapes = [list(p.shape) for p in layer.parameters() if p.requires_grad]
        print(f"Layer {i}: {layer.__class__.__name__}, Trainable parameters: {get_trainable_params(layer)}, Shapes: {shapes}")





def eval_model(
        model,
        lr = 0.001,
        epochs = 5,
) -> List[EvalResult]:
    # Training settings
    batch_size = 256
    test_batch_size = 1000

    kwargs = {'num_workers': 1, 'pin_memory': True}
    transform = transforms.Compose([transforms.ToTensor()])

    # Download and load the training data
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []

    for epoch in range(epochs):
        train(model, train_loader, optimizer, epoch)
        losses.append(test(model, test_loader))

    return losses