import numpy as np
import matplotlib.pyplot as plt

from src.sparse_image_gen import generate_sparse_image
from src.sparse_image_gen import SparseImage
from src.initialize_database import SEMImage
from src.stitch_images import stitch_images


def generate_scan_pattern(lowDTimageObject, sparsityPercent, availableDwellTimes, scanType):
    if not isinstance(lowDTimageObject, SEMImage):
        raise TypeError("First image should be of SEM Object type")
    if sparsityPercent < 0 or sparsityPercent > 100:
        raise ValueError("illegal sparsity percentage")
    if not all(isinstance(dwellTime, int) for dwellTime in availableDwellTimes) or min(availableDwellTimes) < 0:
        raise ValueError("illegal dwell-time")
    if min(availableDwellTimes) < 0:
        raise ValueError("illegal dwell-time")

    sparseImageObject = generate_sparse_image(lowDTimageObject, sparsityPercent, availableDwellTimes)
    imageSize = sparseImageObject.imageSize

    impPixelCoords = sparseImageObject.sparseFeatures[0, :].astype(int)
    if scanType == "descending":
        sortedIntensities = np.argsort(sparseImageObject.sparseFeatures[1, :])[::-1]
        sortedPixelCoords = impPixelCoords[sortedIntensities]
        ycoords = sortedPixelCoords // imageSize
        xcoords = sortedPixelCoords % imageSize
        return ycoords, xcoords

    elif scanType == "descending plus z":
        ycoords = impPixelCoords // imageSize
        xcoords = impPixelCoords % imageSize

        combinedIndices = np.array(list(zip(ycoords, xcoords)))
        sortedPixelCoords = np.array(sorted(combinedIndices, key=lambda x: (-x[0], x[1])))

        return sortedPixelCoords[:, 1], sortedPixelCoords[:, 0]

    elif scanType == "descending plus raster":
        ycoords = impPixelCoords // imageSize
        xcoords = impPixelCoords % imageSize

        combinedIndices = np.array(list(zip(ycoords, xcoords)))
        sortedPixelCoords = combinedIndices[np.lexsort((combinedIndices[:, 1], combinedIndices[:, 0]))]

        return sortedPixelCoords[:, 1], sortedPixelCoords[:, 0]

    else:
        raise ValueError("Invalid scan type")


def display_scan_pattern(lowDTimageObject, sparsityPercent, availableDwellTimes, scanType):
    ycoords, xcoords = generate_scan_pattern(lowDTimageObject, sparsityPercent, availableDwellTimes, scanType)

    plt.figure(figsize=(20, 20))
    plt.title("Path for scanning first 1000 pixels")
    plt.imshow(lowDTimageObject.extractedImage, cmap='grey')
    plt.plot(ycoords[:1000], xcoords[:1000], color='white', linewidth=1)
    plt.show()


def plot_dwell_times_histogram(dwellTimesFeature, bins: int):
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


def display_stitched_image(lowDTImageObject, highDTImageObject, sparsityPercent):
    plt.figure()
    plt.imshow(lowDTImageObject.extractedImage, cmap='grey')
    plt.title("Low DT Image")
    plt.show()

    plt.figure()
    plt.imshow(highDTImageObject.extractedImage, cmap='grey')
    plt.title("High DT Image")
    plt.show()

    stitchedImageNormal = stitch_images(lowDTImageObject, highDTImageObject, sparsityPercent)
    plt.figure()
    plt.title("Normal stitching, dwell-times: {}".format([lowDTImageObject.dwellTime, highDTImageObject.dwellTime]))
    plt.imshow(stitchedImageNormal, cmap='grey')
    plt.show()


def calculate_psnr(originalImage, hybridImage):
    if np.linalg.norm(originalImage - hybridImage) == 0:
        return float('inf')
    return -10 * np.log10(np.mean((originalImage - hybridImage) ** 2))


"""
Execution

from src.initialize_database import read_sem_images
from src.generate_new_images import generate_new_images

path = "D:/Akarsh/Adaptive Scanning/Data/SEM_images_29_May_2024"
availableImages = read_sem_images(path)
imageSubset = availableImages[3:9]
newImageSet = generate_new_images(imageSubset, 4, 10)
imageSubset = sorted(imageSubset + newImageSet, key=lambda eachImage: eachImage.dwellTime)
firstTestImage = imageSubset[0]
secondTestImage = imageSubset[-1]

display_stitched_image(firstTestImage, secondTestImage, 15)
display_scan_pattern(firstTestImage, 15, np.array([10, 30, 40, 50, 100, 200, 300]))
sparseImageObject = generate_sparse_image(firstTestImage, 15, np.array([10, 30, 40, 50, 100, 200, 300]))
display_mask(sparseImageObject, firstTestImage)

display_stitched_image(firstTestImage, secondTestImage, 15, 'gaussian', 3)
plot_dwell_times_histogram(sparseImageObject.sparseFeatures[2, :], 100)
print(compare_stitching_methods(firstTestImage, secondTestImage, 15, 3))

"""