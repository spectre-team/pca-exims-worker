from typing import NamedTuple, Tuple

from kneed import KneeLocator
import numpy as np
import sklearn.decomposition as dec
from spdata.types import Dataset

from ._spectre_adapter import exims


def _pca(dataset: Dataset, n_components: int, backend=dec.PCA, **kwargs) -> \
        Tuple[dec.PCA, np.ndarray]:
    model = backend(n_components=n_components, **kwargs)
    transformed_spectra = model.fit_transform(dataset.spectra)
    return model, transformed_spectra


FeaturesSelection = NamedTuple('FeaturesSelection', [
    ('score', np.ndarray),
    ('index', np.ndarray),
    ('selection', np.ndarray),
    ('threshold', float)
])


def _gradient(values: np.ndarray, order: int=1) -> np.ndarray:
    result = values
    for _ in range(order):
        result = np.gradient(result)
    return result


def _plateau_point(sorted_scores) -> int:
    gradient = np.abs(_gradient(sorted_scores))
    is_plateau = gradient <= np.percentile(gradient, 1)
    plateau_points = np.nonzero(is_plateau)[0]
    return int(np.median(plateau_points))


def _knee_point(decreasing_segment: np.ndarray) -> int:
    locator = KneeLocator(x=np.arange(decreasing_segment.size, dtype=int),
                          y=decreasing_segment,
                          S=1.,
                          invert=False,
                          direction='decreasing')
    assert locator.knee is not None
    return locator.knee


def _select_features(feature_scores) -> FeaturesSelection:
    feature_scores = feature_scores.ravel()
    index = np.argsort(-feature_scores)
    score = feature_scores[index]
    plateau_point = _plateau_point(score)
    concave_up_segment = score[:plateau_point]
    knee_location = _knee_point(concave_up_segment)
    threshold = score[knee_location]
    selection = feature_scores >= threshold
    return FeaturesSelection(score, index, selection, threshold)


ModelledDataset = NamedTuple('ModelledDataset', [
    ('model', dec.PCA),
    ('features', FeaturesSelection),
    ('transformed_dataset', Dataset),
    ('structured_dataset', Dataset)
])


def _dummy_mz(spectra: np.ndarray) -> np.ndarray:
    return np.arange(spectra.shape[1])


def exims_pca(dataset: Dataset, pca_components: int=None, backend=dec.PCA,
              **kwargs) -> ModelledDataset:
    if pca_components is None:
        pca_components = dataset.spectra.shape[1]
    model, transformed_spectra = _pca(dataset, pca_components, backend=backend,
                                      **kwargs)
    transformed_dataset = Dataset(spectra=transformed_spectra,
                                  coordinates=dataset.coordinates,
                                  mz=_dummy_mz(transformed_spectra))
    structness = exims(transformed_dataset)
    features = _select_features(structness)
    structured_features = transformed_spectra[:, features.selection]
    structured_dataset = Dataset(spectra=structured_features,
                                 coordinates=dataset.coordinates,
                                 mz=_dummy_mz(structured_features))
    return ModelledDataset(model, features, transformed_dataset,
                           structured_dataset)
