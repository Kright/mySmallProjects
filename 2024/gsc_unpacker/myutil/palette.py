import glob
from typing import Dict

import numpy as np


class Palette:
    def __init__(self, colors: np.ndarray):
        assert colors.shape == (256, 3), colors.shape
        self.colors = colors

    @staticmethod
    def load(name: str):
        assert name.lower().endswith(".pal")
        colors = np.fromfile(name, dtype=np.uint8)
        assert len(colors) == 768 or np.max(colors[768:]) == 0, f'len(colors) = {len(colors)} for {name}'
        colors = np.reshape(colors[:768], [256, 3])
        return Palette(colors)

    @staticmethod
    def load_all(root: bool) -> Dict[str, 'Palette']:
        return {file: Palette.load(file) for file in glob.glob(f"{root}/*.PAL")}
