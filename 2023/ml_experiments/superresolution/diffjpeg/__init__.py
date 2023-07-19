# all code is from https://github.com/mlomnitz/DiffJPEG under MIT License

from typing import List, Optional
import random
# Pytorch
import torch
import torch.nn as nn
# Local
from .utils import diff_round, quality_to_factor
from .compression import compress_jpeg
from .decompression import decompress_jpeg


class DiffJPEG(nn.Module):
    def __init__(self, height, width, differentiable=True, quality=80):
        ''' Initialize the DiffJPEG layer
        Inputs:
            height(int): Original image hieght
            width(int): Original image width
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standard torch.round
            quality(float): Quality factor for jpeg compression scheme.
        '''
        super(DiffJPEG, self).__init__()
        if differentiable:
            rounding = diff_round
        else:
            rounding = torch.round
        factor = quality_to_factor(quality)
        self.compress = compress_jpeg(rounding=rounding, factor=factor)
        self.decompress = decompress_jpeg(height, width, rounding=rounding,
                                          factor=factor)

    def forward(self, x):
        y, cb, cr = self.compress(x)
        recovered = self.decompress(y, cb, cr)
        return recovered


class RandomJPEG(nn.Module):
    def __init__(self, size: int, qualities: List[int], seed: int = 12):
        super(RandomJPEG, self).__init__()
        self.jpegers = nn.ParameterList(
            [DiffJPEG(height=size, width=size, differentiable=False, quality=q) for q in qualities])
        self.dummy = nn.Parameter(torch.zeros(size=[1]))
        self.random = random.Random(seed)

    def __call__(self, x: torch.Tensor, jpeger_no: Optional[int] = None, each_random: bool = False):
        if each_random and x.dim() == 4 and x.size(0) > 1:
            return torch.cat([self(x[i: i + 1]) for i in range(x.size(0))], dim=0)

        if jpeger_no is None:
            jpeger = self.random.choice(self.jpegers)
        else:
            jpeger = self.jpegers[jpeger_no]

        return jpeger(x.to(self.dummy.device)).to(x.device)
