from typing import List
from torch.utils.data import DataLoader

from .image_dataset import ImagesDataset


class Dataset:
    def __init__(self, roots: List[str], batch_size: int):
        self.roots = roots
        self.images_dataset: ImagesDataset = ImagesDataset.from_dirs_recursive(
            roots,
            shuffle_seed=12,
            channels_order='chw'
        )
        self.cuda_loader = self.make_cuda_loader(batch_size)

    def make_cuda_loader(self, batch_size: int, num_workers: int = 0) -> DataLoader:
        return DataLoader(
            self.images_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            pin_memory_device='cuda',
            num_workers=num_workers
        )


class Datasets:
    def __init__(self):
        self.flowers = Dataset(roots=["datasets/flowers102processed"],
                               batch_size=1)
        self.squares_all = Dataset(roots=["datasets/srgan256_v0", "datasets/srgan256_v1"],
                                   batch_size=1)
