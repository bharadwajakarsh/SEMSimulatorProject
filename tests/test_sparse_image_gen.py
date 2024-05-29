import unittest

import numpy as np

from src.sparse_image_gen import (compute_sample_size, compute_relative_gradient_magnitude,
                                  detect_sharp_edges_indices, detect_high_interest_areas, calculate_pixelwise_dtime)


class TestSparseImageGen(unittest.TestCase):
    def test_sample_size(self):
        image = np.zeros((10, 5), dtype=float)
        n_samples = compute_sample_size(image, 50)
        self.assertEqual(n_samples, 25)

    def test_compute_zero_gradients(self):
        image = np.ones((10, 5), dtype=float)
        gradients = compute_relative_gradient_magnitude(image)
        self.assertEqual(0, np.linalg.norm(gradients - np.zeros(image.shape)))

    def test_compute_nonzero_gradients(self):
        image = np.zeros((5, 5), dtype=float)
        image[0, 0] = 1.0
        image[2, 2] = 1.0
        image[2, 3] = 1.0
        gradients = compute_relative_gradient_magnitude(image)
        expectedGradient = np.asarray(
            [[1, np.sqrt(2) / 4, 0, 0, 0], [np.sqrt(2) / 4, 0, np.sqrt(2) / 4, np.sqrt(2) / 4, 0],
             [0, np.sqrt(2) / 4, np.sqrt(2) / 4, np.sqrt(2) / 4, 1 / np.sqrt(2)],
             [0, 0, np.sqrt(2) / 4, np.sqrt(2) / 4, 0], [0, 0, 0, 0, 0]])
        self.assertAlmostEqual(0, np.linalg.norm(gradients - expectedGradient))

    def test_detect_sharp_edges(self):
        image = np.zeros((5, 5), dtype=float)
        sparsityPercent = 20
        image[0, 0] = 1.0
        image[2, 2] = 1.0
        image[2, 3] = 1.0
        expectedSharpIndices = np.asarray([0, 1, 5, 7, 8])
        gradients = np.asarray(
            [[1, np.sqrt(2) / 4, 0, 0, 0], [np.sqrt(2) / 4, 0, np.sqrt(2) / 4, np.sqrt(2) / 4, 0],
             [0, np.sqrt(2) / 4, np.sqrt(2) / 4, np.sqrt(2) / 4, 1 / np.sqrt(2)],
             [0, 0, np.sqrt(2) / 4, np.sqrt(2) / 4, 0], [0, 0, 0, 0, 0]])
        sharpIndices = detect_sharp_edges_indices(image, gradients, sparsityPercent)
        self.assertEqual(sharpIndices.all(), expectedSharpIndices.all())

    def test_detect_high_interest_areas(self):
        imageGradients = np.asarray(
            [[1, np.sqrt(2) / 4, 0, 0, 0], [np.sqrt(2) / 4, 0, np.sqrt(2) / 4, np.sqrt(2) / 4, 0],
             [0, np.sqrt(2) / 4, np.sqrt(2) / 4, np.sqrt(2) / 4, 1 / np.sqrt(2)],
             [0, 0, np.sqrt(2) / 4, np.sqrt(2) / 4, 0], [0, 0, 0, 0, 0]])
        sharpEdgeIndices = np.asarray([0, 1, 5, 7, 8])
        highInterestAreas = detect_high_interest_areas(imageGradients, sharpEdgeIndices)
        expectedHighAreas = np.asarray([1, np.sqrt(2) / 4, np.sqrt(2) / 4, np.sqrt(2) / 4, np.sqrt(2) / 4])
        self.assertEqual(highInterestAreas.all(), expectedHighAreas.all())

    def test_pixelwise_dtime(self):
        pixelInterests = np.asarray([1, np.sqrt(2) / 4, np.sqrt(2) / 4, np.sqrt(2) / 4, np.sqrt(2) / 4])
        maxDwellTime = 300
        minDwellTime = 0
        dwellTimes = calculate_pixelwise_dtime(pixelInterests, maxDwellTime, minDwellTime)
        expectedDwellTimes = np.asarray([[1, np.sqrt(2) / 4, np.sqrt(2) / 4, np.sqrt(2) / 4, np.sqrt(2) / 4]]) * 300
        self.assertEqual(dwellTimes.all(), expectedDwellTimes.all())


if __name__ == '__main__':
    unittest.main()
