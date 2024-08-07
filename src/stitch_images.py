import numpy as np

from initialize_database import SEMImage, SIMSImage
from sparse_image_gen import extract_sparse_features_sem, extract_sparse_features_sims


def stitch_images_sem(lowDTImageObject, highDTImageObject, sparsityPercent, availableDwellTimes):
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
    imageSize = lowDTImageObject.imageSize

    lowDTFeatures = extract_sparse_features_sem(stitchedImage, sparsityPercent, availableDwellTimes)

    dwellTimesLowDTImage = lowDTFeatures[3]
    yEdgeCoordinates = np.array(lowDTFeatures[0]).astype(int)
    xEdgeCoordinates = np.array(lowDTFeatures[1]).astype(int)

    numberSamples = len(dwellTimesLowDTImage)
    totalSamples = lowDTImageObject.imageSize ** 2

    if np.any(yEdgeCoordinates >= lowDTImageObject.imageSize):
        raise ValueError("Important pixel coordinates out of bounds")
    if np.any(xEdgeCoordinates >= lowDTImageObject.imageSize):
        raise ValueError("Important pixel coordinates out of bounds")

    stitchedImage[yEdgeCoordinates, xEdgeCoordinates] = highDTImage[yEdgeCoordinates, xEdgeCoordinates]
    effectiveDwellTime = (numberSamples * np.mean(dwellTimesLowDTImage)
                          + (totalSamples - numberSamples) * lowDTImageObject.dwellTime) / totalSamples

    return SEMImage(effectiveDwellTime, imageSize, stitchedImage)


def stitch_images_sims(lowDTImageObject, highDTImageObject, sparsityPercent, availableDwellTimes):
    if not isinstance(lowDTImageObject, SIMSImage):
        raise ValueError("Image is not of type SEM Object")
    if not isinstance(highDTImageObject, SIMSImage):
        raise ValueError("Image is not of type SEM Object")
    if lowDTImageObject.dwellTime > highDTImageObject.dwellTime:
        raise ValueError("First image should be of lower dwell-time")
    if lowDTImageObject.extractedImage.shape != highDTImageObject.extractedImage.shape:
        raise ValueError("Images must have the same shape")

    stitchedImageTotal = lowDTImageObject.extractedImage.copy()
    highDTImage = highDTImageObject.extractedImage
    imageSize = lowDTImageObject.imageSize

    stitchedImageSpectrometry = lowDTImageObject.spectrometryImages.copy()
    spectrometryImagesHighDT = highDTImageObject.spectrometryImages

    lowDTFeatures = extract_sparse_features_sims(stitchedImageSpectrometry, sparsityPercent, availableDwellTimes)

    dwellTimesLowDTImage = lowDTFeatures[3]
    yEdgeCoordinates = np.array(lowDTFeatures[0]).astype(int)
    xEdgeCoordinates = np.array(lowDTFeatures[1]).astype(int)

    numberSamples = len(dwellTimesLowDTImage)
    totalSamples = lowDTImageObject.imageSize ** 2

    if np.any(yEdgeCoordinates >= lowDTImageObject.imageSize):
        raise ValueError("Important pixel coordinates out of bounds")
    if np.any(xEdgeCoordinates >= lowDTImageObject.imageSize):
        raise ValueError("Important pixel coordinates out of bounds")

    stitchedImageTotal[yEdgeCoordinates, xEdgeCoordinates] = highDTImage[yEdgeCoordinates, xEdgeCoordinates]

    for i in range(len(stitchedImageSpectrometry)):
        stitchedImageSpectrometry[i][yEdgeCoordinates, xEdgeCoordinates] = spectrometryImagesHighDT[i][yEdgeCoordinates, xEdgeCoordinates]

    effectiveDwellTime = (numberSamples * np.mean(dwellTimesLowDTImage)
                          + (totalSamples - numberSamples) * lowDTImageObject.dwellTime) / totalSamples

    return SIMSImage(imageSize, effectiveDwellTime, stitchedImageSpectrometry, stitchedImageTotal)
