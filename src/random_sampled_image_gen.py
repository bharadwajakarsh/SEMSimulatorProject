import numpy as np
from scipy.interpolate import griddata, LinearNDInterpolator

from src.image_classes import RandomSparseImage


def generate_random_sparse_image_sem(imageObject, sparsityPercent):
    imageToSample = imageObject.extractedImage
    imageSize = imageObject.imageSize
    randomPixelIndices = generate_random_pixel_locations(imageSize, sparsityPercent)
    imageOpened = np.ravel(imageToSample)
    randomPixelIntensities = imageOpened[randomPixelIndices]
    randomSparseFeatures = np.array([randomPixelIndices % imageSize, randomPixelIndices // imageSize,
                                     randomPixelIntensities]).astype(int)

    cornersToAdd = add_corners_to_sample_set(imageToSample)

    for i in range(len(cornersToAdd)):
        if not np.any(np.all(randomSparseFeatures == cornersToAdd[:, i][:, None], axis=0)):
            randomSparseFeatures = np.concatenate((randomSparseFeatures, cornersToAdd[:, i][:, None]), axis=1)

    return RandomSparseImage(randomSparseFeatures, imageSize)


def add_corners_to_sample_set(imageToSample):
    xCornerCoords = np.array([0, 0, len(imageToSample) - 1, len(imageToSample) - 1])
    yCornerCoords = np.array([0, len(imageToSample) - 1, 0, len(imageToSample) - 1])
    cornerPixelIntensities = np.array(
        [imageToSample[xCornerCoords[i], yCornerCoords[i]] for i in range(len(xCornerCoords))])
    return np.array([xCornerCoords, yCornerCoords, cornerPixelIntensities])


def generate_random_pixel_locations(imageSize, sparsityPercent):
    sampleSize = int(imageSize ** 2 * sparsityPercent / 100)
    randomPixelIndices = np.random.choice(imageSize ** 2, size=sampleSize, replace=False)
    return randomPixelIndices


def interpolate_random_sampled_images(randomSparseImage, interpolMethod):
    interpolatedImage = np.zeros((randomSparseImage.imageSize, randomSparseImage.imageSize))
    yCoords, xCoords, pixelIntensities = randomSparseImage.randomSparseFeatures
    points = np.column_stack((xCoords, yCoords))
    gridX, gridY = np.mgrid[0:randomSparseImage.imageSize, 0:randomSparseImage.imageSize]

    if interpolMethod == 'cubic' or 'nearest':
        interpolatedImage = griddata(points, pixelIntensities, (gridX, gridY), interpolMethod)
        interpolatedImage = np.nan_to_num(interpolatedImage, nan=0.0)
    elif interpolMethod == 'natural':
        interpolator = LinearNDInterpolator(points, pixelIntensities)
        interpolatedImage = interpolator(gridX, gridY)
        interpolatedImage = np.nan_to_num(interpolatedImage, nan=0.0)
    else:
        ValueError("Unknown interpolation method")

    return interpolatedImage
