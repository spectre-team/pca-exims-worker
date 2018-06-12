from typing import NamedTuple, Tuple

from kneed import KneeLocator
import numpy as np
import sklearn.decomposition as dec
from spdata.types import Dataset

from ._spectre_adapter import exims


def _pca(dataset: Dataset, n_components: int, backend=dec.IncrementalPCA) -> \
        Tuple[dec.IncrementalPCA, np.ndarray]:
    model = backend(n_components=n_components, batch_size=5 * n_components)
    transformed_spectra = model.fit_transform(dataset.spectra)
    return model, transformed_spectra


FeaturesSelection = NamedTuple('FeaturesSelection', [
    ('score', np.ndarray),
    ('index', np.ndarray),
    ('selection', np.ndarray),
    ('threshold', float)
])


def _inflection_point(sorted_scores) -> int:
    second_derivative = np.diff(sorted_scores, 2)
    neighbour_products = second_derivative[:-1] * second_derivative[1:]
    is_inflection = neighbour_products <= 0
    # +1 because of derivative, +1 because of sign check (array shrinks)
    inflection_point = np.nonzero(is_inflection)[0][0] + 2
    return inflection_point


def _knee_point(decreasing_segment: np.ndarray) -> int:
    locator = KneeLocator(x=np.arange(decreasing_segment.size, dtype=int),
                          y=decreasing_segment,
                          S=1.,
                          invert=False,
                          direction='decreasing')
    return locator.knee


def _select_features(feature_scores) -> FeaturesSelection:
    index = np.argsort(-feature_scores)
    score = feature_scores[index]
    concave_up_segment = score[:_inflection_point(score)]
    knee_location = _knee_point(concave_up_segment)
    threshold = score[knee_location]
    selection = feature_scores >= threshold
    return FeaturesSelection(score, index, selection, threshold)


ModelledDataset = NamedTuple('ModelledDataset', [
    ('model', dec.IncrementalPCA),
    ('features', FeaturesSelection),
    ('transformed_dataset', Dataset),
    ('structured_dataset', Dataset)
])


def _dummy_mz(spectra: np.ndarray) -> np.ndarray:
    return np.arange(spectra.shape[1])


def exims_pca(dataset: Dataset, pca_components: int=None) -> ModelledDataset:
    if pca_components is None:
        pca_components = dataset.spectra.shape[1]
    model, transformed_spectra = _pca(dataset, pca_components)
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
