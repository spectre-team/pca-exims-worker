import unittest

import numpy as np
import numpy.testing as npt

import algorithm._structness as sts


class TestDiscretize(unittest.TestCase):
    def test_finds_several_levels(self):
        image = np.array([
            [1, 1, 5, 6, 2, 6],
            [4, 3, 5, 7, 0, 2],
            [0, 2, 3, 5, 6, 7],
            [0, 1, 3, 4, 7, 4]
        ], dtype=np.uint8) + 5
        all_allowed = np.ones(image.shape, dtype=bool)
        discrete = sts._discretize(image, all_allowed)
        npt.assert_equal(discrete, image-5)


class TestGreycomatrix(unittest.TestCase):
    def test_is_consistent_with_matlab(self):
        image = np.array([
            [1, 1, 5, 6, 2, 6],
            [2, 3, 5, 7, 0, 2],
            [0, 2, 3, 5, 6, 7]
        ], dtype=np.uint8)
        all_allowed = np.ones(image.shape, dtype=bool)
        gcm = sts._greycomatrix(image, all_allowed)
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
        gcm = sts._greycomatrix(image, remove_zero_over_six_and_left_to_two)
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
        darkness, lightness = sts.structness(image)
        self.assertAlmostEqual(darkness, 16)
        self.assertAlmostEqual(lightness, 21)
