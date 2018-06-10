import unittest

import numpy as np
from spdata.types import Coordinates, Dataset

import algorithm


class TestAsImage(unittest.TestCase):
    def test_assigns_values_at_designated_places(self):
        data = np.array([[1], [2], [3], [4]])
        x = np.array([0, 0, 1, 1])
        y = np.array([3, 4, 4, 3])
        image = algorithm.as_image(data, x, y)
        self.assertSequenceEqual(image.shape, (2, 2, 1))
        self.assertEqual(image[0, 0], 1)
        self.assertEqual(image[0, 1], 4)
        self.assertEqual(image[1, 1], 3)
        self.assertEqual(image[1, 0], 2)

    def test_fills_unknown_values_with_predefined(self):
        data = np.array([[1], [2], [3]])
        x = np.array([0, 0, 1])
        y = np.array([3, 4, 4])
        image = algorithm.as_image(data, x, y, default=-1)
        self.assertEqual(image[0, 1], -1)

    def test_creates_channels_for_dimensions(self):
        data = np.array([[1, 5], [2, 6], [3, 7], [4, 8]])
        x = np.array([0, 0, 1, 1])
        y = np.array([3, 4, 4, 3])
        image = algorithm.as_image(data, x, y)
        self.assertSequenceEqual(image.shape, (2, 2, 2))
        self.assertEqual(image[0, 0, 0], 1)
        self.assertEqual(image[0, 0, 1], 5)
        self.assertEqual(image[0, 1, 0], 4)
        self.assertEqual(image[0, 1, 1], 8)
        self.assertEqual(image[1, 1, 0], 3)
        self.assertEqual(image[1, 1, 1], 7)
        self.assertEqual(image[1, 0, 0], 2)
        self.assertEqual(image[1, 0, 1], 6)


class TestExims(unittest.TestCase):
    def test_computes_score_for_each_feature(self):
        coordinates = Coordinates(x=[1, 2, 1], y=[4, 4, 3], z=[0, 0, 0])
        dataset = Dataset(spectra=np.arange(6).reshape(-1, 2),
                          coordinates=coordinates,
                          mz=[8, 9])
        scores = algorithm.exims(dataset)
        self.assertEqual(scores.size, 2)
