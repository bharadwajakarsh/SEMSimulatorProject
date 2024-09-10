import numpy as np

from image_classes import SEMImage, SIMSImage
from sparse_image_gen import extract_sparse_features_sem, extract_sparse_features_sims
from sparse_image_gen import group_features_by_dwell_times


def stitch_images_sem(lowDTImageObject, highDTImageObjects, sparsityPercent):
    if not isinstance(lowDTImageObject, SEMImage):
        raise ValueError("Image is not of type SEM Object")
    for each in highDTImageObjects:
        if not isinstance(each, SEMImage):
            raise ValueError("Image is not of type SEM Object")
        if lowDTImageObject.extractedImage.shape != each.extractedImage.shape:
            raise ValueError("Images must have the same shape")
        if lowDTImageObject.dwellTime > each.dwellTime:
            raise ValueError("First image should be of lower dwell-time")

    availableDwellTimes = [each.dwellTime for each in highDTImageObjects]
    stitchedImage = lowDTImageObject.extractedImage.copy()
    imageSize = lowDTImageObject.imageSize

    lowDTFeatures = extract_sparse_features_sem(stitchedImage, sparsityPercent, availableDwellTimes)
    groupedFeatures = group_features_by_dwell_times(lowDTFeatures)

    for eachUniqueDwellTime in groupedFeatures:
        highDTImage = next(each.extractedImage for each in highDTImageObjects if each.dwellTime == eachUniqueDwellTime)
        yEdgeCoordinates = np.array(groupedFeatures[eachUniqueDwellTime][:, 0]).astype(int)
        xEdgeCoordinates = np.array(groupedFeatures[eachUniqueDwellTime][:, 1]).astype(int)

        if np.any(yEdgeCoordinates >= lowDTImageObject.imageSize):
            raise ValueError("Important pixel coordinates out of bounds")
        if np.any(xEdgeCoordinates >= lowDTImageObject.imageSize):
            raise ValueError("Important pixel coordinates out of bounds")

        stitchedImage[yEdgeCoordinates, xEdgeCoordinates] = highDTImage[yEdgeCoordinates, xEdgeCoordinates]

    effectiveDwellTime = (sparsityPercent / 100) * np.mean(lowDTFeatures[3]) + (
                1 - sparsityPercent / 100) * lowDTImageObject.dwellTime
    return SEMImage(effectiveDwellTime, imageSize, stitchedImage)


def stitch_images_sims(lowDTImageObject, highDTImageObjects, sparsityPercent):
    if not isinstance(lowDTImageObject, SIMSImage):
        raise ValueError("Image is not of type SEM Object")
    for each in highDTImageObjects:
        if not isinstance(each, SIMSImage):
            raise ValueError("Image is not of type SEM Object")
        if lowDTImageObject.extractedImage.shape != each.extractedImage.shape:
            raise ValueError("Images must have the same shape")
        if lowDTImageObject.dwellTime > each.dwellTime:
            raise ValueError("First image should be of lower dwell-time")

    availableDwellTimes = [each.dwellTime for each in highDTImageObjects]
    imageSize = lowDTImageObject.imageSize

    stitchedImageTotal = lowDTImageObject.extractedImage.copy()
    stitchedImageSpectrometry = lowDTImageObject.spectrometryImages.copy()

    lowDTFeatures = extract_sparse_features_sims(stitchedImageSpectrometry, sparsityPercent, availableDwellTimes)
    groupedFeatures = group_features_by_dwell_times(lowDTFeatures)

    for eachUniqueDwellTime in groupedFeatures:
        highDTImage = next(each.extractedImage for each in highDTImageObjects if each.dwellTime == eachUniqueDwellTime)
        spectrometryImagesHighDT = next(each.spectrometryImages for each in highDTImageObjects if
                                        each.dwellTime == eachUniqueDwellTime)
        yEdgeCoordinates = np.array(groupedFeatures[eachUniqueDwellTime][0]).astype(int)
        xEdgeCoordinates = np.array(groupedFeatures[eachUniqueDwellTime][1]).astype(int)

        if np.any(yEdgeCoordinates >= lowDTImageObject.imageSize):
            raise ValueError("Important pixel coordinates out of bounds")
        if np.any(xEdgeCoordinates >= lowDTImageObject.imageSize):
            raise ValueError("Important pixel coordinates out of bounds")

        stitchedImageTotal[yEdgeCoordinates, xEdgeCoordinates] = highDTImage[yEdgeCoordinates, xEdgeCoordinates]
        for i in range(len(stitchedImageSpectrometry)):
            stitchedImageSpectrometry[i][yEdgeCoordinates, xEdgeCoordinates] = spectrometryImagesHighDT[i][yEdgeCoordinates, xEdgeCoordinates]

    effectiveDwellTime = (sparsityPercent / 100) * np.mean(lowDTFeatures[3]) + (
                1 - sparsityPercent / 100) * lowDTImageObject.dwellTime

    return SIMSImage(imageSize, effectiveDwellTime, stitchedImageSpectrometry, stitchedImageTotal)
