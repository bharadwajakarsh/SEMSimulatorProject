import numpy as np
import random
from image_classes import SEMImage


def find_average_image(randomImageSubset):
    if not randomImageSubset:
        raise ValueError("Empty set")

    imageSizeDef = randomImageSubset[0].imageSize

    if len({eachImage.imageSize for eachImage in randomImageSubset}) > 1:
        raise ValueError("Image size not consistent")

    totalImageCount = 0
    dwellTimeSum = 0
    imageSum = np.zeros((imageSizeDef, imageSizeDef))

    for image in randomImageSubset:

        if not isinstance(image, SEMImage):
            print("Image not of type: SEM Image")
            continue

        dwellTimeSum = dwellTimeSum + image.dwellTime
        imageSum = imageSum + image.extractedImage

        totalImageCount = totalImageCount + 1

    return SEMImage(dwellTimeSum / totalImageCount, imageSizeDef, imageSum / totalImageCount)


def generate_new_images(imageSet, subsetSize, numberImages):
    imageSet = [eachImage for eachImage in imageSet if isinstance(eachImage, SEMImage)]

    if not imageSet:
        raise ValueError("Empty set")
    if subsetSize > len(imageSet):
        raise ValueError("Subset size must be smaller than the image set")
    if subsetSize == len(imageSet):
        print("Warning : All images will be identical")

    newImageSubset = []

    for i in range(numberImages):
        randomImageSubset = random.sample(imageSet, subsetSize)
        newImage = find_average_image(randomImageSubset)
        newImageSubset.append(newImage)

    return newImageSubset
