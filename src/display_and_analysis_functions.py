import numpy as np
import matplotlib.pyplot as plt

from src.sparse_image_gen import generate_sparse_image
from src.sparse_image_gen import SparseImage
from src.initialize_database import SEMImage
from src.stitch_images import stitch_images, stitch_with_gaussian_blur


def display_scan_pattern(lowDTimageObject, sparsityPercent, availableDwellTimes):
    if not isinstance(lowDTimageObject, SEMImage):
        raise TypeError("First image should be of SEM Object type")
    if sparsityPercent < 0 or sparsityPercent > 100:
        raise ValueError("illegal sparsity percentage")
    if min(availableDwellTimes) < 0:
        raise ValueError("illegal dwell-time")

    sparseImageObject = generate_sparse_image(lowDTimageObject, sparsityPercent, availableDwellTimes)
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


def display_stitched_image(lowDTImageObject, highDTImageObject, sparsityPercent, stitchType, kernelSize=None):
    if stitchType == 'gaussian':
        if not kernelSize:
            raise ValueError('Must specify kernel size for gaussian-stitching method')
        if not isinstance(kernelSize, int):
            raise ValueError("Kernel size must be an integer.")
        if kernelSize <= 0:
            raise ValueError("Kernel size must be a positive integer.")
        if kernelSize % 2 == 0:
            raise ValueError("Kernel size must be an odd integer.")

    plt.figure()
    plt.imshow(lowDTImageObject.extractedImage, cmap='grey')
    plt.title("Low DT Image")
    plt.show()

    plt.figure()
    plt.imshow(highDTImageObject.extractedImage, cmap='grey')
    plt.title("High DT Image")
    plt.show()

    if stitchType == 'gaussian':
        stitchedImageGauss = stitch_with_gaussian_blur(lowDTImageObject, highDTImageObject, sparsityPercent, kernelSize)
        plt.figure()
        plt.title("Gaussian stitching, dwell-times: {}".format([lowDTImageObject.dwellTime,
                                                                highDTImageObject.dwellTime]))
        plt.imshow(stitchedImageGauss, cmap='grey')
        plt.show()
    elif stitchType == 'normal':
        stitchedImageNormal = stitch_images(lowDTImageObject, highDTImageObject, sparsityPercent)
        plt.figure()
        plt.title("Normal stitching, dwell-times: {}".format([lowDTImageObject.dwellTime, highDTImageObject.dwellTime]))
        plt.imshow(stitchedImageNormal, cmap='grey')
        plt.show()


def calculate_psnr(originalImage, hybridImage):
    if np.linalg.norm(originalImage - hybridImage) == 0:
        return float('inf')
    return -10 * np.log10(np.mean((originalImage - hybridImage) ** 2))


def compare_stitching_methods(lowDTImageObject, highDTImageObject, sparsityPercent, kernelSize):
    lowDTImage = np.copy(lowDTImageObject.extractedImage)
    stitchedImageNormal = stitch_images(lowDTImageObject, highDTImageObject, sparsityPercent)
    stitchedImageGauss = stitch_with_gaussian_blur(lowDTImageObject, highDTImageObject, sparsityPercent, kernelSize)
    psnrNormalStitch = calculate_psnr(lowDTImage, stitchedImageNormal)
    psnrGaussianStitch = calculate_psnr(lowDTImage, stitchedImageGauss)
    return psnrGaussianStitch / psnrNormalStitch


"""
Execution 

from src.initialize_database import refresh_database
from src.generate_new_images import generate_new_images


path = "D:/Akarsh/Adaptive Scanning/Data/SEM_images_29_May_2024"
availableImages = refresh_database(path)
imageSubset = availableImages[3:9]
newImageSet = generate_new_images(imageSubset, 4, 10)
imageSubset = sorted(imageSubset+newImageSet, key=lambda eachImage: eachImage.dwellTime)
firstTestImage = imageSubset[0]
secondTestImage = imageSubset[-1]

display_scan_pattern(firstTestImage, 15, np.array([10, 30, 40, 50, 100, 200, 300]))
sparseImageObject = generate_sparse_image(firstTestImage, 15, np.array([10, 30, 40, 50, 100, 200, 300]))
display_mask(sparseImageObject, firstTestImage)
display_stitched_image(firstTestImage, secondTestImage, 15, 'normal')
display_stitched_image(firstTestImage, secondTestImage, 15, 'gaussian', 3)
plot_dwell_times_histogram(sparseImageObject.sparseFeatures[2, :], 100)
print(compare_stitching_methods(firstTestImage, secondTestImage, 15, 3))

"""