from functools import partial
from typing import Callable, List, Tuple

import numpy as np
from spdata.types import Coordinates, Dataset
from skimage import feature as ft

import matlab_alikes as stats


_DISCRETIZATION_LEVELS = 8
_quantile_thresholds = partial(stats.n_quantiles,
                               N=_DISCRETIZATION_LEVELS - 1)
_greycomatrix_backend = partial(ft.greycomatrix,
                                distances=[1, np.sqrt(2), 1, np.sqrt(2)],
                                angles=np.radians([0., -45., -90., -135.]),
                                symmetric=False,
                                normed=False)


def _greycomatrix(discrete_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    all_allowed = np.all(mask)
    if all_allowed:
        levels = _DISCRETIZATION_LEVELS
    else:
        levels = _DISCRETIZATION_LEVELS + 1
        discrete_image[~mask] = np.max(discrete_image[mask]) + 1
    gcm = _greycomatrix_backend(discrete_image, levels=levels)
    if not all_allowed:
        gcm = gcm[:-1, :-1, :, :]
    return np.dstack(gcm[:, :, i, i] for i in range(gcm.shape[2]))


def _ignorance_mask(image: np.ndarray, ignored: List) -> np.ndarray:
    mask = np.ones(image.shape, dtype=bool)
    for value in ignored:
        mask = np.logical_and(mask, image != value)
    return mask


def _discretize(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image = image.astype(float)
    image /= np.max(image[mask])
    thresholds = np.hstack([_quantile_thresholds(image[mask].ravel()), (1.,)])
    discrete = np.zeros(image.shape, dtype=np.uint8)
    for i in range(len(thresholds)-1, -1, -1):
        pixels_of_interest = np.logical_and(image <= thresholds[i], mask)
        discrete[pixels_of_interest] = i
    return discrete


_split_directions = partial(np.rollaxis, axis=2)
_EXPECTED_BLOCK_SHAPE = 2 * (_DISCRETIZATION_LEVELS / 2,)


def _block_structness(greycomatrix_block: np.ndarray) -> float:
    if greycomatrix_block.shape != _EXPECTED_BLOCK_SHAPE:
        raise ValueError(f'Expected shape {_EXPECTED_BLOCK_SHAPE}, '
                         f'got {greycomatrix_block.shape}.')
    return np.sum(greycomatrix_block[:3, :3]) \
        + np.sum(greycomatrix_block[:2, :2]) \
        + 2. * greycomatrix_block[0, 0]


_BlockSelector = Callable[[np.ndarray], np.ndarray]


def _darkness(greycomatrix: np.ndarray) -> np.ndarray:
    size = int(_DISCRETIZATION_LEVELS / 2)
    return greycomatrix[:size, :size]


def _lightness(greycomatrix: np.ndarray) -> np.ndarray:
    size = int(_DISCRETIZATION_LEVELS / 2 + 1)
    return greycomatrix[:-size:-1, :-size:-1]


def _structness_of(selector: _BlockSelector, directions: np.ndarray) -> float:
    return float(np.sum(
        _block_structness(selector(direction))
        for direction in directions
    ))


def structness(image: np.ndarray, ignored: List=None) -> Tuple[float, float]:
    if ignored is None:
        ignored = []
    mask = _ignorance_mask(image, ignored)
    discrete = _discretize(image, mask)
    greycomatrix = _greycomatrix(discrete, mask)
    directions = _split_directions(greycomatrix)
    structness_of_darkness = _structness_of(_darkness, directions)
    structness_of_lightness = _structness_of(_lightness, directions)
    return structness_of_darkness, structness_of_lightness


def as_image(data: np.ndarray, x: np.ndarray, y: np.ndarray, default=-1) -> \
        np.ndarray:
    translated_x, translated_y = x - np.min(x), y - np.min(y)
    rows, columns = np.max(translated_y) + 1, np.max(translated_x) + 1
    cube = default * np.ones((rows, columns, data.shape[1]))
    cube[translated_y, translated_x] = data
    return cube
