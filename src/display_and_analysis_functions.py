import numpy as np
import matplotlib.pyplot as plt

from src.sparse_image_gen import extract_sparse_features
from src.sparse_image_gen import SparseImage
from src.initialize_database import SEMImage
from src.stitch_images import stitch_images


def group_features_by_dwell_times(sparseFeatures):
    columnIndex = 2
    uniqueDwellTimes = np.unique(sparseFeatures[columnIndex])
    groupedSparseFeatures = {value: [] for value in uniqueDwellTimes}

    for eachDwellTime in uniqueDwellTimes:
        mask = sparseFeatures[columnIndex] == eachDwellTime
        featuresOfGroup = sparseFeatures[:, mask]
        groupedSparseFeatures[eachDwellTime] = featuresOfGroup

    return groupedSparseFeatures


def generate_scan_pattern(lowDTimageObject, sparsityPercent, availableDwellTimes, scanType):
    if not isinstance(lowDTimageObject, SEMImage):
        raise TypeError("First image should be of SEM Object type")
    if sparsityPercent < 0 or sparsityPercent > 100:
        raise ValueError("illegal sparsity percentage")
    if min(availableDwellTimes) < 0:
        raise ValueError("illegal dwell-time")

    ourImage = lowDTimageObject.extractedImage
    imageSize = lowDTimageObject.imageSize

    sparseFeatures = extract_sparse_features(ourImage, sparsityPercent, availableDwellTimes)
    groupedSparseFeatures = group_features_by_dwell_times(sparseFeatures)

    if scanType == "ascending":
        xImportantPixels = sparseFeatures[0, :].astype(int)
        yImportantPixels = sparseFeatures[1, :].astype(int)
        sortedIntensities = np.argsort(sparseFeatures[2, :])
        ycoords = yImportantPixels[sortedIntensities]
        xcoords = xImportantPixels[sortedIntensities]
        return ycoords, xcoords

    elif scanType == "ascending plus z":
        groupedPixelLocations = {}
        for eachUniqueDwellTime in groupedSparseFeatures:
            xImportantPixels = np.array(groupedSparseFeatures[eachUniqueDwellTime][0, :]).astype(int)
            yImportantPixels = np.array(groupedSparseFeatures[eachUniqueDwellTime][1, :]).astype(int)

            combinedIndices = np.array(list(zip(yImportantPixels, xImportantPixels)))
            sortedPixelCoords = np.array(sorted(combinedIndices, key=lambda x: (x[0], x[1])))
            groupedPixelLocations[eachUniqueDwellTime] = sortedPixelCoords

        return groupedPixelLocations

    elif scanType == "ascending plus raster":
        groupedPixelLocations = {}
        for eachUniqueDwellTime in groupedSparseFeatures:
            xImportantPixels = np.array(groupedSparseFeatures[eachUniqueDwellTime][0, :]).astype(int)
            yImportantPixels = np.array(groupedSparseFeatures[eachUniqueDwellTime][1, :]).astype(int)
            combinedIndices = np.array(list(zip(yImportantPixels, xImportantPixels)))
            sortedPixelCoords = combinedIndices[np.lexsort((combinedIndices[:, 1], combinedIndices[:, 0]))]
            groupedPixelLocations[eachUniqueDwellTime] = sortedPixelCoords

        return groupedPixelLocations

    else:
        raise ValueError("Invalid scan type")


def display_scan_pattern(lowDTimageObject, sparsityPercent, availableDwellTimes, scanType):
    if scanType == "ascending":
        ycoords, xcoords = generate_scan_pattern(lowDTimageObject, sparsityPercent, availableDwellTimes, scanType)
        plt.figure(figsize=(20, 20))
        plt.title("Path for scanning first 1000 pixels")
        plt.imshow(lowDTimageObject.extractedImage, cmap='grey')
        plt.plot(ycoords[:1000], xcoords[:1000], color='white', linewidth=1)
        plt.show()

    elif scanType == "ascending plus z" or "ascending plus raster":
        groupedPixelLocations = generate_scan_pattern(lowDTimageObject, sparsityPercent, availableDwellTimes, scanType)
        for i, eachUniqueDwellTime in enumerate(groupedPixelLocations):
            ycoords = groupedPixelLocations[eachUniqueDwellTime][:, 1]
            xcoords = groupedPixelLocations[eachUniqueDwellTime][:, 0]
            plt.figure(figsize=(20, 20))
            plt.title(f"Path for scan number {i+1}")
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
    xPixelLocations = sparseImageObject.sparseFeatures[0, :].astype(int)
    yPixelLocations = sparseImageObject.sparseFeatures[1, :].astype(int)

    imageToSee[yPixelLocations, xPixelLocations] = sparseImageObject.sparseFeatures[2, :]
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


"""
Execution

def calculate_psnr(originalImage, hybridImage):
    if np.linalg.norm(originalImage - hybridImage) == 0:
        return float('inf')
    return -10 * np.log10(np.mean((originalImage - hybridImage) ** 2))

from src.initialize_database import read_sem_images
from src.generate_new_images import generate_new_images

path = "D:/Akarsh/Adaptive Scanning/Data/SEM_images_29_May_2024"
availableImages = read_sem_images(path)
imageSubset = availableImages[3:9]
newImageSet = generate_new_images(imageSubset, 4, 10)
imageSubset = sorted(imageSubset + newImageSet, key=lambda eachImage: eachImage.dwellTime)
firstTestImage = imageSubset[0]
secondTestImage = imageSubset[-1]

display_scan_pattern(firstTestImage, 15, np.array([10, 30, 40, 50, 100, 200, 300]), "ascending")

display_scan_pattern(firstTestImage, 15, np.array([10, 30, 40, 50, 100, 200, 300]), "ascending")
sparseImageObject = generate_sparse_image(firstTestImage, 15, np.array([10, 30, 40, 50, 100, 200, 300]))
display_mask(sparseImageObject, firstTestImage)
display_stitched_image(firstTestImage, secondTestImage, 15)
display_stitched_image(firstTestImage, secondTestImage, 15, 'gaussian', 3)
plot_dwell_times_histogram(sparseImageObject.sparseFeatures[2, :], 100)
print(compare_stitching_methods(firstTestImage, secondTestImage, 15, 3))

"""
