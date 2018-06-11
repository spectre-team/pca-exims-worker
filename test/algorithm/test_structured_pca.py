import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from spdata.types import Coordinates, Dataset

import algorithm._structured_pca as spca


def returns(value) -> MagicMock:
    return MagicMock(return_value=value)


simulated_structness = np.array([
    1000, 400, 200, 100, 75, 60, 55, 50, 45, 35, 20, 5
])


class TestInflectionPoint(unittest.TestCase):
    def test_finds_plateau(self):
        inflection = spca._inflection_point(simulated_structness)
        self.assertLessEqual(inflection, 9)
        self.assertGreaterEqual(inflection, 4)


class TestKneePoint(unittest.TestCase):
    def test_finds_knee_point(self):
        knee = spca._knee_point(simulated_structness[:-3])
        self.assertGreaterEqual(knee, 2)
        self.assertLessEqual(knee, simulated_structness.size - 6)


class TestSelectFeatures(unittest.TestCase):
    def test_selects_some_features(self):
        selection = spca._select_features(simulated_structness)
        self.assertGreaterEqual(selection.selection.sum(), 2)
        self.assertLess(selection.selection.sum(), selection.selection.size)


class TestEximsPca(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        features_number = simulated_structness.size
        some_spectra = np.random.randn(100, features_number)
        dummy_coordinates = Coordinates(*(3 * [np.arange(100)]))
        self.dataset = Dataset(some_spectra,
                               dummy_coordinates,
                               mz=np.arange(features_number))

    def test_transforms_dataset_and_picks_structured_components(self):
        with patch.object(spca, spca.exims.__name__, returns(
                simulated_structness)):
            modelled = spca.exims_pca(self.dataset)
        self.assertLess(modelled.structured_dataset.spectra.shape[1],
                        self.dataset.spectra.shape[1])
        self.assertEqual(modelled.structured_dataset.spectra.shape[0],
                         self.dataset.spectra.shape[0])
