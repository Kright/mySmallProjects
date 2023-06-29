from typing import Optional
import numpy as np


class CovarianceAccumulator:
    def __init__(self):
        self.samples_count: int = 0
        self.sum: Optional[np.ndarray] = None
        self.prod_xy: Optional[np.ndarray] = None

    @property
    def average(self) -> np.ndarray:
        return self.sum / self.samples_count

    @property
    def covariance(self) -> np.ndarray:
        av = self.average
        return (self.prod_xy / self.samples_count) - av[:, np.newaxis] * av[np.newaxis, :]

    @property
    def covariance_eigen_values(self) -> np.ndarray:
        eig = np.linalg.eig(self.covariance)[0]
        return np.sort(eig)[::-1]

    def add_sample(self, vec: np.ndarray):
        assert isinstance(vec, np.ndarray)
        assert len(vec.shape) == 1
        vec_size = vec.shape[0]

        self.samples_count += 1
        self._add_sum(vec)
        self._add_prod(vec.reshape([1, vec_size]) * vec.reshape([vec_size, 1]))

    def add_samples(self, vecs: np.ndarray):
        assert isinstance(vecs, np.ndarray)
        assert len(vecs.shape) == 2, f'Unexpected shape {vecs.shape}'
        count, vec_size = vecs.shape

        self.samples_count += count
        self._add_sum(vecs.sum(axis=0))
        matrices = vecs[:, :, np.newaxis] * vecs[:, np.newaxis, :]
        self._add_prod(matrix=np.sum(matrices, axis=0))

    def _add_sum(self, vec: np.ndarray):
        if self.sum is None:
            self.sum = np.zeros(shape=vec.shape, dtype=np.double)
        self.sum += vec

    def _add_prod(self, matrix: np.ndarray):
        if self.prod_xy is None:
            self.prod_xy = np.zeros(shape=matrix.shape, dtype=np.double)
        self.prod_xy += matrix