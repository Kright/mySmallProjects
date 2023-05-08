import os
from pathlib import Path
from typing import Optional, List, NoReturn

import numpy as np
from PIL import Image as PILImage


def compare_shape(array: np.ndarray, shape: List[int]) -> bool:
    if len(array.shape) != len(shape):
        return False

    for elem, expected in zip(array.shape, shape):
        if expected is None:
            continue
        if elem != expected:
            return False

    return True


def assert_shape(array: np.ndarray, shape: List[int]) -> NoReturn:
    assert compare_shape(array, shape), f"wrong shape, got {array.shape}, expected {shape}"


def get_extension(path: str) -> str:
    return Path(path).suffix.lower()


def is_image(path: str) -> bool:
    return get_extension(path) in {'.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}


def image_get_model(image: PILImage) -> str:
    model_tag = 0x0110
    return image.getexif().get(model_tag)


def downsample_with_gamma(image: np.ndarray, factor: int, gamma: float = 2.2) -> np.ndarray:
    assert_shape(image, [None, None, 3])

    h, w, c = image.shape
    if h % factor != 0 or w % factor != 0:
        image = image[0: h // factor * factor, 0: w // factor * factor]

    assert np.issubdtype(image.dtype, np.floating)
    linear = image ** gamma
    downsampled = linear.reshape((image.shape[0] // factor, factor, image.shape[1] // factor, factor, 3)).mean(3).mean(1)
    assert_shape(downsampled, [image.shape[0] // factor, image.shape[1] // factor, 3])
    return downsampled ** (1.0 / gamma)
