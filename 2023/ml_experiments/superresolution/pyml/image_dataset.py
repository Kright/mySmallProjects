import logging
import os
from random import Random
from typing import List, Optional, Set

import PIL
import PIL.ImageOps
import numpy as np
import torch.utils.data
from PIL import Image as PILImage

from .image import is_image, assert_shape


class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self,
                 images_paths: List[str],
                 channels_order: str = 'hwc'):
        assert channels_order in {'hwc', 'chw'}
        super(ImagesDataset).__init__()
        self.channels_order = channels_order
        self.images_paths: List[str] = images_paths

    def __getitem__(self, index) -> Optional[np.ndarray]:
        path = self.images_paths[index]
        try:
            pil_image = PILImage.open(path)
            pil_image = PIL.ImageOps.exif_transpose(pil_image)
            pil_image = pil_image.convert('RGB')

            np_array = np.asarray(pil_image)
            assert_shape(np_array, [None, None, 3])

            np_array = self.to_float_img(np_array, path)

            if self.channels_order == 'chw':
                np_array = np.moveaxis(np_array, 2, 0)

            return np_array
        except Exception as e:
            logging.error(f"error {e} for {path}")
            return None

    def to_float_img(self, array: np.ndarray, path: str) -> np.ndarray:
        assert array.dtype == np.uint8, f"unknown type = {array.dtype} and shape {array.shape} at {path}"
        return array.astype(np.float32) / 255.0

    def __len__(self):
        return len(self.images_paths)

    def __str__(self):
        return f"MyImagesDataset(len = {len(self)})"

    def __repr__(self):
        return str(self)

    @staticmethod
    def from_dirs_recursive(roots: List[str], shuffle_seed: Optional[int] = None, channels_order: str = 'hwc'):
        result_set: Set[str] = set()

        for root in roots:
            for dir_path, dir_names, file_names in os.walk(root):
                for file_name in file_names:
                    if is_image(file_name):
                        result_set.add(f"{dir_path}/{file_name}")

        results = list(result_set)
        results.sort()
        if shuffle_seed is not None:
            Random(shuffle_seed).shuffle(results)

        return ImagesDataset(results, channels_order)
