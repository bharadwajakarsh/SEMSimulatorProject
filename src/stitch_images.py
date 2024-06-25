import numpy as np

from src.initialize_database import SEMImage
from src.sparse_image_gen import (compute_image_of_relative_gradients, detect_sharp_edges_indices,
                                  calculate_pixel_interests)


def get_numpy_gaussian_kernel(kernelSize, sigma):
    kernel = np.exp(-(np.arange(kernelSize) - kernelSize // 2) ** 2 / (2 * sigma ** 2))
    return kernel / np.sum(kernel)


def stitch_images(lowDTImageObject, highDTImageObject, sparsityPercent):
    if not isinstance(lowDTImageObject, SEMImage):
        raise ValueError("Image is not of type SEM Object")
    if not isinstance(highDTImageObject, SEMImage):
        raise ValueError("Image is not of type SEM Object")
    if lowDTImageObject.dwellTime > highDTImageObject.dwellTime:
        raise ValueError("First image should be of lower dwell-time")
    if lowDTImageObject.extractedImage.shape != highDTImageObject.extractedImage.shape:
        raise ValueError("Images must have the same shape")

    stitchedImage = lowDTImageObject.extractedImage.copy()
    highDTImage = highDTImageObject.extractedImage

    gradientsLowDTImage = compute_image_of_relative_gradients(stitchedImage)
    impPixelCoords = detect_sharp_edges_indices(gradientsLowDTImage, sparsityPercent)

    stitchedImageFlat = stitchedImage.ravel()
    highDTImageFlat = highDTImage.ravel()

    if np.any(impPixelCoords >= stitchedImageFlat.size):
        raise ValueError("Important pixel coordinates out of bounds")

    stitchedImageFlat[impPixelCoords] = highDTImageFlat[impPixelCoords]

    return stitchedImageFlat.reshape(lowDTImageObject.extractedImage.shape)

