from functools import partial
from typing import Callable, Tuple

import numpy as np
from functional import pipe, progress_bar, for_each
from spdata.types import Coordinates, Dataset

from ._structness import structness


def as_image(data: np.ndarray, x: np.ndarray, y: np.ndarray, default=-1) -> \
        np.ndarray:
    x, y = x.astype(int), y.astype(int)
    translated_x, translated_y = x - np.min(x), y - np.min(y)
    rows, columns = int(np.max(translated_y) + 1), int(np.max(translated_x) + 1)
    if len(data.shape) < 2:
        data = data.reshape((data.shape[0], 1))
    cube = default * np.ones((rows, columns, data.shape[1]))
    cube[translated_y, translated_x] = data
    return cube


_IGNORED = -1
_Feature = np.ndarray
_Structness = float
_FeatureProcessor = Callable[[_Feature], Tuple[_Structness, _Structness]]
_remove_channel_dimension = partial(np.squeeze, axis=2)


def _feature_processor(coordinates: Coordinates) -> _FeatureProcessor:
    # noinspection PyTypeChecker
    return pipe(
        partial(as_image, x=coordinates.x, y=coordinates.y, default=_IGNORED),
        _remove_channel_dimension,
        partial(structness, ignored=[_IGNORED])
    )


def _normalize_columns(matrix) -> np.ndarray:
    matrix = np.array(matrix, dtype=float)
    matrix += np.finfo(float).eps
    assert len(matrix.shape) == 2
    return matrix / np.max(matrix, axis=0)


_as_features = np.transpose
_normalize_structness_by_kind = _normalize_columns
_sumarize_structness_by_feature = pipe(partial(np.sum, axis=1), np.ravel)
FeaturesStructness = np.ndarray
_StructnessEstimator = Callable[[Dataset], FeaturesStructness]


def _estimator(structness_: _FeatureProcessor) -> _StructnessEstimator:
    # noinspection PyTypeChecker
    return pipe(
        _as_features,
        progress_bar('feature structness'),
        for_each(structness_, parallel=True),
        _normalize_structness_by_kind,
        _sumarize_structness_by_feature
    )


def exims(data: Dataset) -> FeaturesStructness:
    structness_estimator = _estimator(_feature_processor(data.coordinates))
    return structness_estimator(data.spectra)
