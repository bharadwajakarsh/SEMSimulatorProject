import numpy as np
import matplotlib.pyplot as plt

from src.sparse_image_gen import generate_sparse_image
from src.sparse_image_gen import SparseImage
from src.intialize_database import SEMImage


def display_scan_pattern(lowDTimageObject, sparsityPercent, maxDwellTime, minDwellTime):
    if not isinstance(lowDTimageObject, SEMImage):
        raise TypeError("First image should be of SEM Object type")
    if sparsityPercent < 0 or sparsityPercent > 100:
        raise ValueError("illegal sparsity percentage")
    if minDwellTime < 0:
        raise ValueError("illegal dwell-time")
    if maxDwellTime <= minDwellTime:
        raise ValueError("Max dwell time should be greater than min dwell time")

    sparseImageObject = generate_sparse_image(lowDTimageObject, sparsityPercent, maxDwellTime, minDwellTime)
    imgSize = sparseImageObject.imageSize

    impPixelCoords = sparseImageObject.sparseFeatures[0, :].astype(int)
    sortedIntensities = np.argsort(sparseImageObject.sparseFeatures[1, :])[::-1]

    sortedPixelCoords = impPixelCoords[sortedIntensities]

    xcoords = sortedPixelCoords // imgSize
    ycoords = sortedPixelCoords % imgSize

    plt.figure(figsize=(20, 20))
    plt.title("Path for scanning first 1000 pixels")
    plt.imshow(lowDTimageObject.extractedImage, cmap='grey')
    plt.plot(xcoords[:1000], ycoords[:1000], color='white', linewidth=1)
    plt.show()


def dwell_times_histogram(dwellTimesFeature, bins: int):
    if len(dwellTimesFeature) == 0:
        raise ValueError("Empty dwell-times feature vector")
    if not isinstance(bins, int):
        bins = int(bins)

    plt.figure()
    plt.hist(dwellTimesFeature, bins)
    plt.xlabel("dwell time(us)")
    plt.ylabel("# pixels")
    plt.show()


def display_mask(sparseImageObject: SparseImage, originalImageObject: SEMImage):
    if not isinstance(sparseImageObject, SparseImage):
        raise TypeError("Input should be a 'Sparse Image' object")
    if not isinstance(originalImageObject, SEMImage):
        raise TypeError("Input should be a 'SEM Image' object")

    imageToSee = np.zeros(originalImageObject.extractedImage.size)
    imageToSee[sparseImageObject.sparseFeatures[0, :].astype(int)] = sparseImageObject.sparseFeatures[1, :]
    imageToSee = np.reshape(imageToSee, originalImageObject.extractedImage.shape)

    plt.figure()
    plt.title('Original image')
    plt.imshow(originalImageObject.extractedImage, cmap='grey')
    plt.show()
    plt.figure()
    plt.title('Mask of HIA (negative)')
    plt.imshow(1 - imageToSee, cmap='grey')
    plt.show()


"""
Execution

path = "C:/Users/akbh02/JupyterNotebooks"
availableImages = refresh_database(path)
testImage = generate_new_images(availableImages, 10, 10)[5]
display_scan_pattern(testImage, 15, 300, 10)

"""
