import numpy as np
from PIL import Image
from .palette import Palette
from typing import Optional


class Screen:
    def __init__(self, w: int, h: int, fill_color: int = 0):
        self.x = 0
        self.y = 0
        self.x1 = w
        self.y1 = h
        self.data = np.zeros(shape=[self.height * self.width], dtype=np.uint16)
        self.data_2d = np.reshape(self.data, (self.height, self.width))
        self.fill(fill_color)

    @property
    def width(self) -> int:
        return self.x1 - self.x

    @property
    def height(self) -> int:
        return self.y1 - self.y

    def fill(self, color: int):
        self.data.fill(color)

    def as_pic(self, palette: Optional[Palette] = None) -> Image.Image:
        return Screen.make_pic(self.data_2d, palette)

    def as_interesting_pic(self, palette: Optional[Palette] = None, rm_top_left: bool = True) -> Optional[Image.Image]:
        pic = self.data_2d

        if rm_top_left:
            while pic.shape[0] > 0 and np.max(pic[0]) == 0:
                pic = pic[1:]
        while pic.shape[0] > 0 and np.max(pic[-1]) == 0:
            pic = pic[:-1]

        if pic.shape[0] == 0:
            return None

        if rm_top_left:
            while pic.shape[1] > 0 and np.max(pic[:, 0]) == 0:
                pic = pic[:, 1:]
        while pic.shape[1] > 1 and np.max(pic[:, -1]) == 0:
            pic = pic[:, :-1]

        if pic.shape[1] == 0:
            return None

        return Screen.make_pic(pic, palette)

    @staticmethod
    def make_pic(pic: np.ndarray, palette: Optional[Palette]) -> Image.Image:
        if palette is None:
            return Image.fromarray(pic, mode="L")

        flat_indices = np.reshape(pic, [-1])
        rgb_pic = palette.colors[flat_indices].reshape((pic.shape[0], pic.shape[1], 4))
        return Image.fromarray(rgb_pic, mode="RGBA")
