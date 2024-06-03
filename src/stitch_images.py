import numpy as np
import cv2

from src.intialize_database import SEMImage
from src.sparse_image_gen import (compute_image_of_relative_gradients, detect_sharp_edges_indices,
                                  detect_high_interest_areas)


def calculate_psnr(originalImage, hybridImage):
    if np.linalg.norm(originalImage - hybridImage) == 0:
        return float('inf')
    return -10 * np.log10(np.mean((originalImage - hybridImage) ** 2))


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

    return np.reshape(stitchedImage, lowDTImageObject.extractedImage.shape)


def stitch_with_gaussian_blur(lowDTImageObject, highDTImageObject, sparsityPercent, kernelSize):
    stitchedImage = np.copy(lowDTImageObject.extractedImage)
    highDTImage = np.copy(highDTImageObject.extractedImage)

    gradientsLowDTImage = compute_image_of_relative_gradients(stitchedImage)
    impPixelCoords = detect_sharp_edges_indices(gradientsLowDTImage, sparsityPercent)

    kernelOneD = cv2.getGaussianKernel(kernelSize[0], 0)
    kernelTwoD = np.outer(kernelOneD, kernelOneD.T)

    maskToSee = np.zeros(stitchedImage.size)
    maskToSee[impPixelCoords] = detect_high_interest_areas(gradientsLowDTImage, impPixelCoords)
    maskToSee = np.reshape(maskToSee, lowDTImageObject.extractedImage.shape)

    blurdMask = np.convolve(np.ravel(maskToSee), np.ravel(kernelTwoD), mode='same').reshape(maskToSee.shape)

    for i in range(len(blurdMask)):
        for j in range(len(blurdMask)):
            if blurdMask[i, j] != 0:
                stitchedImage[i, j] = highDTImage[i, j]

    return stitchedImage
