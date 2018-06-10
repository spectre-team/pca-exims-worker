import unittest

import numpy as np
import numpy.testing as npt

import algorithm


class TestDiscretize(unittest.TestCase):
    def test_finds_several_levels(self):
        image = np.array([
            [1, 1, 5, 6, 2, 6],
            [4, 3, 5, 7, 0, 2],
            [0, 2, 3, 5, 6, 7],
            [0, 1, 3, 4, 7, 4]
        ], dtype=np.uint8) + 5
        all_allowed = np.ones(image.shape, dtype=bool)
        discrete = algorithm._discretize(image, all_allowed)
        npt.assert_equal(discrete, image-5)


class TestGreycomatrix(unittest.TestCase):
    def test_is_consistent_with_matlab(self):
        image = np.array([
            [1, 1, 5, 6, 2, 6],
            [2, 3, 5, 7, 0, 2],
            [0, 2, 3, 5, 6, 7]
        ], dtype=np.uint8)
        all_allowed = np.ones(image.shape, dtype=bool)
        gcm = algorithm._greycomatrix(image, all_allowed)
        twos_right_to_zeros = 2
        self.assertEqual(gcm[0, 2, 0], twos_right_to_zeros)
        threes_right_up_zeros = 1
        self.assertEqual(gcm[0, 3, 1], threes_right_up_zeros)
        zeros_above_six = 1
        self.assertEqual(gcm[6, 0, 2], zeros_above_six)
        fives_left_up_fives = 1
        self.assertEqual(gcm[5, 5, 3], fives_left_up_fives)

    def test_allows_masking(self):
        image = np.array([
            [1, 1, 5, 6, 2, 6],
            [2, 3, 5, 7, 0, 2],
            [0, 2, 3, 5, 6, 7]
        ], dtype=np.uint8)
        remove_zero_over_six_and_left_to_two = np.ones(image.shape, dtype=bool)
        remove_zero_over_six_and_left_to_two[1, -2] = False
        gcm = algorithm._greycomatrix(image, remove_zero_over_six_and_left_to_two)
        twos_right_to_zeros = 1
        self.assertEqual(gcm[0, 2, 0], twos_right_to_zeros)
        threes_right_up_zeros = 1
        self.assertEqual(gcm[0, 3, 1], threes_right_up_zeros)
        zeros_above_six = 0
        self.assertEqual(gcm[6, 0, 2], zeros_above_six)
        fives_left_up_fives = 1
        self.assertEqual(gcm[5, 5, 3], fives_left_up_fives)


class TestStructness(unittest.TestCase):
    def test_computes_structness_of_image(self):
        image = np.array([
            [1, 1, 5, 6, 2, 6],
            [4, 3, 5, 7, 0, 2],
            [0, 2, 3, 5, 6, 7],
            [0, 1, 3, 4, 7, 4]
        ], dtype=np.uint8)
        darkness, lightness = algorithm.structness(image)
        self.assertAlmostEqual(darkness, 16)
        self.assertAlmostEqual(lightness, 21)


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
