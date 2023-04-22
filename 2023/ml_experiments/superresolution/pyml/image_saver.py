import os
from typing import Optional

import numpy as np
from PIL import Image as PILImage

from .image import assert_shape


class ImageSaver:
    def __init__(self, path: str, subdir_size: Optional[int] = None):
        os.makedirs(path, exist_ok=True)
        self.path: str = path
        self.saved_count: int = 0
        self.subdir_size: Optional[int] = subdir_size

    def saveHWC(self, image: np.ndarray, name_prefix: str = ''):
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image, 0.0, 1.0)
            image = (image * 255.0).astype(np.uint8)
        assert_shape(image, [None, None, 3])
        assert image.dtype == np.uint8
        self.savePIL(PILImage.fromarray(image), name_prefix)

    def saveCHW(self, image: np.ndarray, name_prefix: str = ''):
        assert_shape(image, [3, None, None])
        self.saveHWC(np.moveaxis(image, 0, 2), name_prefix)

    def saveBCHW(self, batch: np.ndarray, name_prefix: str = ''):
        assert_shape(batch, [None, 3, None, None])
        for image in batch:
            self.saveCHW(image, name_prefix)

    def savePIL(self, image: PILImage, name_prefix: str = ''):
        if self.subdir_size is None:
            dir_path = self.path
        else:
            subdir = self.saved_count // self.subdir_size
            dir_path = f"{self.path}/{subdir}"
            os.makedirs(dir_path, exist_ok=True)

        image_path = f"{dir_path}/{name_prefix}{self.saved_count}.png"
        image.save(image_path)
        self.saved_count += 1
