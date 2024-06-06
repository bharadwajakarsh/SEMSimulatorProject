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

    stitchedImage = lowDTImageObject.extractedImage
    highDTImage = highDTImageObject.extractedImage

    gradientsLowDTImage = compute_image_of_relative_gradients(stitchedImage)
    impPixelCoords = detect_sharp_edges_indices(gradientsLowDTImage, sparsityPercent)

    stitchedImageFlat = np.ravel(stitchedImage)
    highDTImageFlat = np.ravel(highDTImage)

    stitchedImageFlat[impPixelCoords] = highDTImageFlat[impPixelCoords]

    return np.reshape(stitchedImageFlat, lowDTImageObject.extractedImage.shape)


def stitch_with_gaussian_blur(lowDTImageObject, highDTImageObject, sparsityPercent, kernelSize):
    if not isinstance(lowDTImageObject, SEMImage):
        raise ValueError("Image is not of type SEM Object")
    if not isinstance(highDTImageObject, SEMImage):
        raise ValueError("Image is not of type SEM Object")
    if lowDTImageObject.dwellTime > highDTImageObject.dwellTime:
        raise ValueError("First image should be of lower dwell-time")
    if sparsityPercent < 0 or sparsityPercent > 100:
        raise ValueError("illegal sparsity percentage")

    stitchedImage = np.copy(lowDTImageObject.extractedImage)
    highDTImage = highDTImageObject.extractedImage

    gradientsLowDTImage = compute_image_of_relative_gradients(stitchedImage)
    impPixelCoords = detect_sharp_edges_indices(gradientsLowDTImage, sparsityPercent)

    kernelOneD = get_numpy_gaussian_kernel(kernelSize, 0.1)
    kernelTwoD = np.outer(kernelOneD, kernelOneD.T)
    kernelTwoDFlat = np.ravel(kernelTwoD)

    maskToSee = np.zeros(stitchedImage.size)
    maskToSee[impPixelCoords] = calculate_pixel_interests(gradientsLowDTImage, impPixelCoords)

    blurdMask = np.convolve(maskToSee, kernelTwoDFlat, mode='same').reshape(stitchedImage.shape)

    for i in range(len(blurdMask)):
        for j in range(len(blurdMask)):
            if blurdMask[i, j] != 0:
                stitchedImage[i, j] = highDTImage[i, j]

    return stitchedImage
