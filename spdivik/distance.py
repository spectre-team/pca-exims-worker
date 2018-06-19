"""
distance.py
Common interface for distance metric

Copyright 2018 Spectre Team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from abc import ABCMeta, abstractmethod
from enum import Enum
import numpy as np
import scipy.spatial.distance as dist


class DistanceMetric(object, metaclass=ABCMeta):
    """Measures distance between points in multidimensional space"""
    @abstractmethod
    def _intradistance(self, matrix2d: np.ndarray) -> np.ndarray:
        """Compute distances between all pairs of points in the matrix

        @param matrix2d: 2D matrix with points in rows
        @return: 2D matrix with distances between points.
        result[i, j] describes distance from matrix2d[i] to matrix2d[j]
        """
        pass

    @abstractmethod
    def _interdistance(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        """Compute distances between all pairs of points between matrices

        @param first: 2D matrix with points in rows
        @param second: 2D matrix with points in rows
        @return: 2D matrix with distances between points.
        result[i, j] describes distance from first[i] to second[j]
        """
        pass

    def __call__(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        """Compute distances between points

        Distances between points in all pairs between both matrices.

        result[i, j] describes distance from first[i] to second[j]

        @param first: 2D matrix with points in rows
        @param second: 2D matrix with points in rows
        @return: 2D matrix with distances between points
        """
        if not isinstance(first, np.ndarray) or len(first.shape) != 2:
            raise ValueError("first matrix must be 2D np.ndarray")
        if not isinstance(second, np.ndarray) or len(second.shape) != 2:
            raise ValueError("second matrix must be 2D np.ndarray")
        if second is first:
            distances = self._intradistance(first)
        else:
            distances = self._interdistance(first, second)
        message = self.__class__.__name__ + "breaks distance metric contract"
        assert isinstance(distances, np.ndarray), message
        assert len(distances.shape) == 2, message
        assert distances.shape[0] == first.shape[0], message
        assert distances.shape[1] == second.shape[0], message
        return distances


class KnownMetric(Enum):
    braycurtis = "braycurtis"
    canberra = "canberra"
    chebyshev = "chebyshev"
    cityblock = "cityblock"
    correlation = "correlation"
    cosine = "cosine"
    dice = "dice"
    euclidean = "euclidean"
    hamming = "hamming"
    jaccard = "jaccard"
    kulsinski = "kulsinski"
    mahalanobis = "mahalanobis"
    atching = "atching"
    minkowski = "minkowski"
    rogerstanimoto = "rogerstanimoto"
    russellrao = "russellrao"
    sokalmichener = "sokalmichener"
    sokalsneath = "sokalsneath"
    sqeuclidean = "sqeuclidean"
    yule = "yule"


class ScipyDistance(DistanceMetric):
    """DistanceMetric based on scipy distances"""
    def __init__(self, name: KnownMetric, **kwargs):
        """
        @param name: name of the metric used
        @param kwargs: optional arguments specified in scipy for that
        specific metric
        """
        self._name = name.value
        self._optionals = kwargs

    def _intradistance(self, matrix2d: np.ndarray) -> np.ndarray:
        """Compute distances between all pairs (pdist)

        @param matrix2d: 2D matrix with points in rows
        @return: 2D matrix of pairwise distances
        """
        vector = dist.pdist(matrix2d, metric=self._name, **self._optionals)
        return dist.squareform(vector)

    def _interdistance(self, first: np.ndarray, second: np.ndarray) -> \
            np.ndarray:
        """Compute distances between all pairs of points between arrays (cdist)

        @param first: 2D matrix with points in rows
        @param second: 2D matrix with points in rows
        @return: 2D matrix of pairwise distances
        """
        return dist.cdist(first, second, metric=self._name, **self._optionals)
