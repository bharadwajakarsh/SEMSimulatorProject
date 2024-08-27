import numpy as np
from skimage import filters

from src.image_classes import SparseImageSIMS, SparseImageSEM


def compute_sample_size(imageShape, sparsityPercent):
    return int(imageShape[0] * imageShape[1] * sparsityPercent / 100)


def compute_image_of_relative_gradients(image):
    relativeGradientsImage = np.asarray(filters.sobel(image))
    maxGradient = np.max(relativeGradientsImage)

    if maxGradient != 0.0:
        return relativeGradientsImage / maxGradient

    return relativeGradientsImage


def detect_sharp_edge_locations(relativeGradientsImage, sparsityPercent):
    threshold = np.percentile(relativeGradientsImage, 100 - sparsityPercent)
    MaskOfSharpPixels = relativeGradientsImage >= threshold
    return np.where(MaskOfSharpPixels)


def calculate_pixel_interests(relativeGradientsImage, ySharpIndices, xSharpIndices):
    if any(y < 0 or y >= relativeGradientsImage.shape[0] for y in ySharpIndices):
        raise ValueError("Index value out of range")
    if any(x < 0 or x >= relativeGradientsImage.shape[1] for x in xSharpIndices):
        raise ValueError("Index value out of range")
    return relativeGradientsImage[ySharpIndices, xSharpIndices]


def calculate_pixelwise_dtime(pixelInterests, availableDwellTimes):
    normalizedPixelInterests = (pixelInterests - np.min(pixelInterests)) / (
            np.max(pixelInterests) - np.min(pixelInterests))
    maxDwellTime = max(availableDwellTimes)
    minDwellTime = min(availableDwellTimes)
    dwellTimes = minDwellTime + normalizedPixelInterests * (maxDwellTime - minDwellTime)
    return np.asarray([min(availableDwellTimes, key=lambda x: abs(x - dtime)) for dtime in dwellTimes])


def extract_sparse_features_sem(extractedImage, sparsityPercent, availableDwellTimes):
    relativeGradientsImage = compute_image_of_relative_gradients(extractedImage)
    ySharpIndices, xSharpIndices = detect_sharp_edge_locations(relativeGradientsImage, sparsityPercent)
    pixelInterests = calculate_pixel_interests(relativeGradientsImage, ySharpIndices, xSharpIndices)

    if max(pixelInterests) == 0:
        raise RuntimeError("Useless Image. No edges present")

    estDwellTime = calculate_pixelwise_dtime(pixelInterests, availableDwellTimes)

    return np.array([ySharpIndices, xSharpIndices, pixelInterests, estDwellTime])


def extract_sparse_features_sims(spectrometryImages, sparsityPercent, availableDwellTimes):
    sumImage = np.zeros(spectrometryImages[0].shape)

    for eachMassImage in spectrometryImages:
        sumImage = sumImage + eachMassImage

    sumImageNormalized = sumImage/np.max(sumImage)

    ySharpIndices, xSharpIndices = detect_sharp_edge_locations(sumImageNormalized, sparsityPercent)
    pixelInterests = calculate_pixel_interests(sumImageNormalized, ySharpIndices, xSharpIndices)

    if max(pixelInterests) == 0:
        raise RuntimeError("Useless Image")

    estDwellTime = calculate_pixelwise_dtime(pixelInterests, availableDwellTimes)

    return np.array([ySharpIndices, xSharpIndices, pixelInterests, estDwellTime])


def generate_sparse_image_sem(imageObject, sparsityPercent, availableDwellTimes):
    if sparsityPercent < 0 or sparsityPercent > 100:
        raise ValueError("illegal sparsity percentage")

    imageSizeDef = imageObject.imageSize
    ourImage = imageObject.extractedImage
    sparseFeaturesSEM = extract_sparse_features_sem(ourImage, sparsityPercent, availableDwellTimes)

    return SparseImageSEM(sparseFeaturesSEM, imageSizeDef)


def generate_sparse_image_sims(imageObject, sparsityPercent, availableDwellTimes):
    if sparsityPercent < 0 or sparsityPercent > 100:
        raise ValueError("illegal sparsity percentage")

    imageSizeDef = imageObject.imageSize
    spectrometryImages = imageObject.spectrometryImages
    sparseFeaturesSIMS = extract_sparse_features_sims(spectrometryImages, sparsityPercent, availableDwellTimes)

    return SparseImageSIMS(sparseFeaturesSIMS, imageSizeDef)


def group_features_by_dwell_times(sparseFeatures):
    columnIndex = 3
    uniqueDwellTimes = np.unique(sparseFeatures[columnIndex])
    groupedSparseFeatures = {value: [] for value in uniqueDwellTimes}

    for eachDwellTime in uniqueDwellTimes:
        mask = sparseFeatures[columnIndex] == eachDwellTime
        featuresOfGroup = sparseFeatures[:, mask]
        groupedSparseFeatures[eachDwellTime] = featuresOfGroup

    return groupedSparseFeatures
