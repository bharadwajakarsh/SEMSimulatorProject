import numpy as np
from scipy.interpolate import griddata, LinearNDInterpolator


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
