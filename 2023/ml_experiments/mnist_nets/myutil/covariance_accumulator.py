from typing import Optional, Tuple

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
    def eigenvalues(self) -> np.ndarray:
        eig = np.linalg.eig(self.covariance)[0]
        return np.sort(eig)[::-1]

    @property
    def eigenvectors(self) -> np.ndarray:
        eigenvalues, eigenvectors = np.linalg.eig(self.covariance)
        order = np.argsort(eigenvalues)[::-1]
        return eigenvectors[:, order]

    def to_eigenvalues_and_back(self, components_count: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        matrix = self.eigenvectors
        return self.average, matrix[:, 0:components_count], np.linalg.inv(matrix)[0:components_count, :]

    @property
    def eigenvalues_normalized(self) -> np.ndarray:
        eig = self.eigenvalues
        return eig / np.sum(eig)

    def add_sample(self, vec: np.ndarray) -> 'CovarianceAccumulator':
        assert isinstance(vec, np.ndarray)
        assert len(vec.shape) == 1
        vec_size = vec.shape[0]

        self.samples_count += 1
        self._add_sum(vec)
        self._add_prod(vec.reshape([1, vec_size]) * vec.reshape([vec_size, 1]))

        return self

    def add_samples(self, arr: np.ndarray, axis: int = -1, max_size: int = 100_000_000) -> 'CovarianceAccumulator':
        assert isinstance(arr, np.ndarray)
        assert len(arr.shape) >= 2, f'Unexpected shape {arr.shape}'

        arr = np.moveaxis(arr, axis, -1)
        if len(arr.shape) > 2:
            arr = np.reshape(arr, [-1, arr.shape[-1]])
        arr = arr.astype(np.double, copy=False)

        self.samples_count += arr.shape[0]
        self._add_sum(arr.sum(axis=0))

        max_batch_size = max(1, max_size // (arr.shape[1] ** 2))

        for offset in range(0, arr.shape[0], max_batch_size):
            upper_bound = min(arr.shape[0], offset + max_batch_size)
            part = arr[offset:upper_bound]
            matrices = part[:, :, np.newaxis] * part[:, np.newaxis, :]
            self._add_prod(matrix=np.sum(matrices, axis=0))

        return self

    def _add_sum(self, vec: np.ndarray):
        if self.sum is None:
            self.sum = np.zeros(shape=vec.shape, dtype=np.double)
        self.sum += vec

    def _add_prod(self, matrix: np.ndarray):
        if self.prod_xy is None:
            self.prod_xy = np.zeros(shape=matrix.shape, dtype=np.double)
        self.prod_xy += matrix
