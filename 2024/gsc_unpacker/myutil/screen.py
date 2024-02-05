import numpy as np
from PIL import Image
from .palette import Palette
from typing import Optional


class Screen:
    def __init__(self, w: int, h: int):
        self.x = 0
        self.y = 0
        self.x1 = w
        self.y1 = h
        self.data = np.zeros(shape=[self.height * self.width], dtype=np.uint8)
        self.data_2d = np.reshape(self.data, (self.height, self.width))

    @property
    def width(self) -> int:
        return self.x1 - self.x

    @property
    def height(self) -> int:
        return self.y1 - self.y

    def as_pic(self, palette: Optional[Palette] = None) -> Image:
        return Screen.make_pic(self.data_2d, palette)

    def as_interesting_pic(self, palette: Optional[Palette] = None) -> Image:
        pic = self.data_2d

        while np.max(pic[0]) == 0 and pic.shape[0] > 1:
            pic = pic[1:]
        while np.max(pic[-1]) == 0 and pic.shape[0] > 1:
            pic = pic[:-1]
        while np.max(pic[:, 0]) == 0 and pic.shape[1] > 1:
            pic = pic[:, 1:]
        while np.max(pic[:, -1]) == 0 and pic.shape[1] > 1:
            pic = pic[:, :-1]

        return Screen.make_pic(pic, palette)

    @staticmethod
    def make_pic(pic: np.ndarray, palette: Optional[Palette]) -> Image:
        if palette is None:
            return Image.fromarray(pic, mode="L")

        flat_indices = np.reshape(pic, [-1])
        rgb_pic = palette.colors[flat_indices].reshape((pic.shape[0], pic.shape[1], 3))
        return Image.fromarray(rgb_pic, mode="RGB")
