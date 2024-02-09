import glob
from typing import Dict

import numpy as np


class Palette:
    TRANSPARENT = 256

    def __init__(self, colors: np.ndarray):
        assert colors.shape == (257, 4), colors.shape
        self.colors = colors
        # 257 is transparent color

    @staticmethod
    def load(name: str):
        assert name.lower().endswith(".pal")
        colors = np.fromfile(name, dtype=np.uint8)
        assert len(colors) == 768 or np.max(colors[768:]) == 0, f'len(colors) = {len(colors)} for {name}'
        colors = np.reshape(colors[:768], [256, 3])

        argb_colors = np.zeros((257, 4), dtype=np.uint8)
        argb_colors[:256, 0:3] = colors
        argb_colors[:256, 3] = 255

        return Palette(argb_colors)

    @staticmethod
    def load_all(root: bool) -> Dict[str, 'Palette']:
        return {file: Palette.load(file) for file in glob.glob(f"{root}/*.PAL")}
