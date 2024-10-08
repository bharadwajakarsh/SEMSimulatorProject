import unittest

import numpy as np

from src.analysis import calculate_psnr
from src.image_classes import SEMImage
from src.stitch_images import stitch_images_sem


class TestStitchImages(unittest.TestCase):

    def test_stitch_images(self):
        exampleLDTImage = SEMImage(20, 5, np.zeros((5, 5), dtype=float))
        exampleLDTImage.extractedImage[0, 1] = 20
        exampleLDTImage.extractedImage[3, 1] = 5
        exampleLDTImage.extractedImage[1, 2] = 10
        exampleLDTImage.extractedImage[4, 0] = 2
        exampleLDTImage.extractedImage[3, 3] = 165
        exampleHDTImage = SEMImage(200, 5, np.ones((5, 5), dtype=float))

        expectedStitchedImage = np.array(
            [[0, 20, 0, 0, 0], [0, 0, 10, 0, 0], [0, 0, 0, 1, 0], [0, 5, 0, 165, 1], [2, 0, 0, 1, 0]]).astype(float)
        stitchedImageForTest = stitch_images_sem(exampleLDTImage, exampleHDTImage, 15)

        self.assertEqual(0, np.linalg.norm(expectedStitchedImage - stitchedImageForTest))

    def test_zero_mse_psnr(self):
        firstImage = np.ones((5, 5))
        secondImage = np.ones((5, 5))
        psnrValue = calculate_psnr(originalHDTImage, secondImage)

        self.assertAlmostEqual(psnrValue, float('inf'))

    def test_nonzero_mse_psnr(self):
        firstImage = np.ones((5, 5))*0.1
        secondImage = np.zeros((5, 5))
        psnrValue = calculate_psnr(originalHDTImage, secondImage)

        self.assertEqual(psnrValue, 20.0)


if __name__ == '__main__':
    unittest.main()