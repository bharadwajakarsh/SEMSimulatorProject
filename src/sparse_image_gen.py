import numpy as np
from skimage import filters

from image_classes import SparseImage, SEMImage, SIMSImage


def compute_sample_size(imageShape, sparsityPercent):
    return int(imageShape[0] * imageShape[1] * sparsityPercent / 100)


def compute_image_of_relative_gradients(image):
    relativeGradientsImage = np.asarray(filters.sobel(image))
    maxGradient = np.max(relativeGradientsImage)

    if maxGradient != 0.0:
        return relativeGradientsImage / maxGradient

    return relativeGradientsImage


def detect_sharp_edge_locations(image, sparsityPercent):
    threshold = np.percentile(image, 100 - sparsityPercent)
    MaskOfSharpPixels = image >= threshold
    return np.where(MaskOfSharpPixels)


def calculate_pixel_interests(image, ySharpIndices, xSharpIndices):
    if any(y < 0 or y >= image.shape[0] for y in ySharpIndices):
        raise ValueError("Index value out of range")
    if any(x < 0 or x >= image.shape[1] for x in xSharpIndices):
        raise ValueError("Index value out of range")
    return image[ySharpIndices, xSharpIndices]


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


def generate_sparse_image(imageObject, sparsityPercent, sparseImageType, availableDwellTimes=None):
    imageSize = imageObject.imageSize
    extractedImage = imageObject.extractedImage

    if sparsityPercent < 0 or sparsityPercent > 100:
        raise ValueError("illegal sparsity percentage")

    if sparseImageType == 'hia':
        if isinstance(imageObject, SEMImage):
            sparseFeaturesSEM = extract_sparse_features_sem(extractedImage, sparsityPercent, availableDwellTimes)
            return SparseImage(sparseImageType, imageSize, sparseFeaturesSEM)

        elif isinstance(imageObject, SIMSImage):
            spectrometryImages = imageObject.spectrometryImages
            sparseFeaturesSIMS = extract_sparse_features_sims(spectrometryImages, sparsityPercent, availableDwellTimes)
            return SparseImage(sparseImageType, imageSize, sparseFeaturesSIMS)
        else:
            raise ValueError("Unknown image type")

    elif sparseImageType == 'random':

        randomPixelIndices = generate_random_pixel_locations(imageSize, sparsityPercent)
        imageOpened = np.ravel(extractedImage)
        randomPixelIntensities = imageOpened[randomPixelIndices]
        randomSparseFeatures = np.array([randomPixelIndices % imageSize, randomPixelIndices // imageSize,
                                         randomPixelIntensities]).astype(int)
        cornersToAdd = add_corners_to_sample_set(extractedImage)
        for i in range(len(cornersToAdd)):
            if not np.any(np.all(randomSparseFeatures == cornersToAdd[:, i][:, None], axis=0)):
                randomSparseFeatures = np.concatenate((randomSparseFeatures, cornersToAdd[:, i][:, None]), axis=1)

        return SparseImage(sparseImageType, imageSize, randomSparseFeatures)
    else:
        raise ValueError("Unknown type")


def group_features_by_dwell_times(sparseFeatures):
    columnIndex = 3
    uniqueDwellTimes = np.unique(sparseFeatures[columnIndex])
    groupedSparseFeatures = {value: [] for value in uniqueDwellTimes}

    for eachDwellTime in uniqueDwellTimes:
        mask = sparseFeatures[columnIndex] == eachDwellTime
        featuresOfGroup = sparseFeatures[:, mask]
        groupedSparseFeatures[eachDwellTime] = featuresOfGroup

    return groupedSparseFeatures


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
