import unittest

import numpy as np

from src.sparse_image_gen import (compute_sample_size, compute_image_of_relative_gradients,
                                  detect_sharp_edge_locations, calculate_pixel_interests, calculate_pixelwise_dtime)


class TestSparseImageGen(unittest.TestCase):
    def test_sample_size(self):
        imageShape = (10, 5)
        n_samples = compute_sample_size(imageShape, 50)
        self.assertEqual(n_samples, 25)

    def test_compute_zero_gradients(self):
        image = np.ones((10, 5), dtype=float)
        gradients = compute_image_of_relative_gradients(image)
        self.assertEqual(0, np.linalg.norm(gradients - np.zeros(image.shape)))

    def test_compute_nonzero_gradients(self):
        image = np.zeros((5, 5), dtype=float)
        image[0, 0] = 1.0
        image[2, 2] = 1.0
        image[2, 3] = 1.0
        gradients = compute_image_of_relative_gradients(image)
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
        expectedYSharpIndices = np.asarray([1, 1, 0, 2, 0])
        expectedXSharpIndices = np.asarray([2, 0, 1, 4, 0])
        gradients = np.asarray(
            [[1, np.sqrt(2) / 4, 0, 0, 0], [np.sqrt(2) / 4, 0, np.sqrt(2) / 4, np.sqrt(2) / 4, 0],
             [0, np.sqrt(2) / 4, np.sqrt(2) / 4, np.sqrt(2) / 4, 1 / np.sqrt(2)],
             [0, 0, np.sqrt(2) / 4, np.sqrt(2) / 4, 0], [0, 0, 0, 0, 0]])
        ySharpIndices, xSharpIndices = detect_sharp_edge_locations(gradients, sparsityPercent)
        self.assertEqual(0, np.linalg.norm(xSharpIndices - expectedXSharpIndices))
        self.assertEqual(0, np.linalg.norm(ySharpIndices - expectedYSharpIndices))

    def test_detect_high_interest_areas(self):
        imageGradients = np.asarray(
            [[1, np.sqrt(2) / 4, 0, 0, 0], [np.sqrt(2) / 4, 0, np.sqrt(2) / 4, np.sqrt(2) / 4, 0],
             [0, np.sqrt(2) / 4, np.sqrt(2) / 4, np.sqrt(2) / 4, 1 / np.sqrt(2)],
             [0, 0, np.sqrt(2) / 4, np.sqrt(2) / 4, 0], [0, 0, 0, 0, 0]])
        ySharpEdgeIndices = np.asarray([1, 1, 0, 2, 0])
        xSharpEdgeIndices = np.asarray([2, 0, 1, 4, 0])
        highInterestAreas = calculate_pixel_interests(imageGradients, ySharpEdgeIndices, xSharpEdgeIndices)
        expectedHighAreas = np.asarray([np.sqrt(2) / 4, np.sqrt(2) / 4, np.sqrt(2) / 4, 1/np.sqrt(2), 1])
        self.assertEqual(0, np.linalg.norm(highInterestAreas - expectedHighAreas))

    def test_pixelwise_dtime(self):
        pixelInterests = np.asarray([1, np.sqrt(2) / 4, np.sqrt(2) / 4, np.sqrt(2) / 4, np.sqrt(2) / 4])
        availableDwellTimes = np.asarray([10, 30, 40, 50, 100, 200, 300])
        dwellTimes = calculate_pixelwise_dtime(pixelInterests, availableDwellTimes)
        expectedDwellTimes = np.asarray([[300, 10, 10, 10, 10]])
        self.assertEqual(0, np.linalg.norm(dwellTimes - expectedDwellTimes))


if __name__ == '__main__':
    unittest.main()
