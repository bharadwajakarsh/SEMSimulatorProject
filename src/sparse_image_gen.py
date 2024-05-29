import numpy as np


class SparseImage:
    def __init__(self, sparseFeatures, imageSize):
        self.sparseFeatures = sparseFeatures
        self.imageSize = imageSize


def compute_sample_size(image, sparsityPercent):
    if sparsityPercent < 0 or sparsityPercent > 100:
        raise ValueError("illegal sparsity percentage")
    return int(image.size * sparsityPercent / 100)


def compute_relative_gradient_magnitude(image):
    gradients_x = np.gradient(image, axis=1)
    gradients_y = np.gradient(image, axis=0)
    imageGradients = np.sqrt(gradients_x ** 2 + gradients_y ** 2)
    if np.max(imageGradients) != 0.0:
        return imageGradients / np.max(imageGradients)
    else:
        return imageGradients


def detect_sharp_edges_indices(image, imageGradients, sparsityPercent):
    sampleSize = compute_sample_size(image, sparsityPercent)
    threshold = np.partition(np.ravel(imageGradients), -sampleSize)[-sampleSize]
    sharpEdgesIndices = np.where(np.ravel(imageGradients) >= threshold)[0][:sampleSize]
    return sharpEdgesIndices


def detect_high_interest_areas(imageGradients, sharpEdgeIndices):
    if any(x < 0 or x > imageGradients.size for x in sharpEdgeIndices):
        raise ValueError("Index value out of range")
    return np.ravel(imageGradients)[sharpEdgeIndices]


def calculate_pixelwise_dtime(pixelInterests, maxDwellTime, minDwellTime):
    if max(pixelInterests) == 0:
        raise RuntimeError("Useless Image. No edges present")
    if maxDwellTime == minDwellTime:
        raise ValueError("Invalid range for dwell-time")
    return minDwellTime + pixelInterests * (maxDwellTime - minDwellTime)


def extract_sparse_features(extractedImage, sparsityPercent, maxDwellTime, minDwellTime):
    imageGradients = compute_relative_gradient_magnitude(extractedImage)
    sharpEdgesIndices = detect_sharp_edges_indices(extractedImage, imageGradients, sparsityPercent)
    pixelInterests = detect_high_interest_areas(imageGradients, sharpEdgesIndices)
    estDwellTime = calculate_pixelwise_dtime(pixelInterests, maxDwellTime, minDwellTime)
    return np.array([sharpEdgesIndices, pixelInterests, estDwellTime])


def generate_sparse_image(imageObject, samplePercent, maxDwellTime, minDwellTime):
    imageSizeDef = imageObject.imageSize
    ourImage = imageObject.extractedImage
    sparseFeatures = extract_sparse_features(ourImage, samplePercent, maxDwellTime, minDwellTime)
    return SparseImage(sparseFeatures, imageSizeDef)
