import numpy as np

from src.initialize_database import SEMImage
from src.sparse_image_gen import (compute_image_of_relative_gradients, detect_sharp_edge_locations,
                                  calculate_pixel_interests)


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
    yEdgeCoordinates, xEdgeCoordinates = detect_sharp_edge_locations(gradientsLowDTImage, sparsityPercent)

    if np.any(yEdgeCoordinates >= lowDTImageObject.imageSize):
        raise ValueError("Important pixel coordinates out of bounds")
    if np.any(xEdgeCoordinates >= lowDTImageObject.imageSize):
        raise ValueError("Important pixel coordinates out of bounds")

    stitchedImage[yEdgeCoordinates, xEdgeCoordinates] = highDTImage[yEdgeCoordinates, xEdgeCoordinates]

    return stitchedImage
