import numpy as np


class SparseImage:
    def __init__(self, sparseFeatures, imageSize):
        self.sparseFeatures = sparseFeatures
        self.imageSize = imageSize


def compute_sample_size(imageShape, sparsityPercent):
    return int(imageShape[0]*imageShape[1] * sparsityPercent / 100)


def compute_image_of_relative_gradients(image):
    gradients_x = np.gradient(image, axis=1)
    gradients_y = np.gradient(image, axis=0)
    relativeGradientsImage = np.sqrt(gradients_x ** 2 + gradients_y ** 2)
    maxGradient = np.max(relativeGradientsImage)

    if maxGradient != 0.0:
        return relativeGradientsImage/maxGradient

    return relativeGradientsImage


def detect_sharp_edges_indices(relativeGradientsImage, sparsityPercent):
    sampleSize = compute_sample_size(relativeGradientsImage.shape, sparsityPercent)
    relativeGradientsFlat = np.ravel(relativeGradientsImage)

    threshold = np.partition(relativeGradientsFlat, -sampleSize)[-sampleSize]
    sharpEdgesIndices = np.where(relativeGradientsFlat >= threshold)[0][:sampleSize]
    return sharpEdgesIndices


def detect_high_interest_areas(relativeGradientsImage, sharpEdgeIndices):
    if any(x < 0 or x >= relativeGradientsImage.size for x in sharpEdgeIndices):
        raise ValueError("Index value out of range")
    return np.ravel(relativeGradientsImage)[sharpEdgeIndices]


def calculate_pixelwise_dtime(pixelInterests, maxDwellTime, minDwellTime):
    return minDwellTime + pixelInterests * (maxDwellTime - minDwellTime)


def extract_sparse_features(extractedImage, sparsityPercent, maxDwellTime, minDwellTime):
    relativeGradientsImage = compute_image_of_relative_gradients(extractedImage)
    sharpEdgesIndices = detect_sharp_edges_indices(relativeGradientsImage, sparsityPercent)
    pixelInterests = detect_high_interest_areas(relativeGradientsImage, sharpEdgesIndices)

    if max(pixelInterests) == 0:
        raise RuntimeError("Useless Image. No edges present")

    estDwellTime = calculate_pixelwise_dtime(pixelInterests, maxDwellTime, minDwellTime)

    return np.array([sharpEdgesIndices, pixelInterests, estDwellTime])


def generate_sparse_image(imageObject, sparsityPercent, maxDwellTime, minDwellTime):
    if sparsityPercent < 0 or sparsityPercent > 100:
        raise ValueError("illegal sparsity percentage")
    if maxDwellTime <= minDwellTime:
        raise ValueError("Invalid range for dwell-time")

    imageSizeDef = imageObject.imageSize
    ourImage = imageObject.extractedImage
    sparseFeatures = extract_sparse_features(ourImage, sparsityPercent, maxDwellTime, minDwellTime)

    return SparseImage(sparseFeatures, imageSizeDef)
