import unittest
import numpy as np

from src.generate_new_images import generate_new_images, find_average_image
from src.intialize_database import SEMImage


class TestNewImageGeneration(unittest.TestCase):

    def test_average_image(self):
        testImageSet = [SEMImage(100, 5, np.zeros((5, 5))),
                        SEMImage(30, 5, np.ones((5, 5))),
                        SEMImage(50, 5, np.zeros((5, 5)))]
        testAverageImage = find_average_image(testImageSet)
        expectedImage = SEMImage(60, 5, np.ones((5, 5)) * (1 / 3))

        self.assertEqual(testAverageImage.dwellTime, expectedImage.dwellTime)
        self.assertEqual(0, np.linalg.norm(testAverageImage.extractedImage - expectedImage.extractedImage))
        self.assertEqual(testAverageImage.imageSize, expectedImage.imageSize)

    def test_generate_images(self):
        testImageSet = [SEMImage(100, 5, np.zeros((5, 5))),
                        SEMImage(30, 5, np.ones((5, 5))),
                        SEMImage(50, 5, np.zeros((5, 5)))]

        newImageSet = generate_new_images(testImageSet, 3, 2)
        expectedSet = [SEMImage(60, 5, np.ones((5, 5)) * (1 / 3)),
                       SEMImage(60, 5, np.ones((5, 5)) * (1 / 3))]

        for i in range(len(expectedSet)):
            self.assertEqual(newImageSet[i].imageSize, expectedSet[i].imageSize)
            self.assertEqual(newImageSet[i].dwellTime, expectedSet[i].dwellTime)
            self.assertEqual(0, np.linalg.norm(newImageSet[i].extractedImage - expectedSet[i].extractedImage))

    def test_anomalous_image(self):
        testImageSet = [SEMImage(100, 5, np.zeros((5, 5))),
                        SEMImage(30, 5, np.ones((5, 5))),
                        SEMImage(50, 5, np.zeros((5, 5))), np.ones((4, 4))]

        newImageSet = generate_new_images(testImageSet, 3, 2)
        expectedSet = [SEMImage(60, 5, np.ones((5, 5)) * (1 / 3)),
                       SEMImage(60, 5, np.ones((5, 5)) * (1 / 3))]

        for i in range(len(expectedSet)):
            self.assertEqual(newImageSet[i].imageSize, expectedSet[i].imageSize)
            self.assertEqual(newImageSet[i].dwellTime, expectedSet[i].dwellTime)
            self.assertEqual(0, np.linalg.norm(newImageSet[i].extractedImage - expectedSet[i].extractedImage))


if __name__ == '__main__':
    unittest.main()
