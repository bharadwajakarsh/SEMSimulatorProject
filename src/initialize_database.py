import pandas as pd
import numpy as np
import glob
import os

from PIL import Image


class SEMImage:
    def __init__(self, dwellTime, imageSize, extractedImage):
        self.dwellTime = dwellTime
        self.imageSize = imageSize
        self.extractedImage = extractedImage


class SIMSImage:
    def __init__(self, imageSize, dwellTime, spectrometryImages, extractedImage):
        self.imageSize = imageSize
        self.dwellTime = dwellTime
        self.spectrometryImages = spectrometryImages
        self.extractedImage = extractedImage


def read_sem_images(folderPath):
    imageSet = []
    csvFiles = glob.glob(os.path.join(folderPath, '*.csv'))
    if not csvFiles:
        raise TypeError("No csv file found in folder")
    for path in csvFiles:
        try:
            imageDataFrame = pd.read_csv(path, delimiter=';', dtype=str)
            dwellTime = float(imageDataFrame.iloc[1, 0])
            imageSize = int(imageDataFrame.iloc[1, 1])
            extractedImage = np.asarray(imageDataFrame.iloc[3:, :].astype(float))

            if dwellTime < 0 or dwellTime > 400:
                raise ValueError("Dwell time out of range")
            if imageSize <= 0:
                raise ValueError("Invalid Image size")
            if extractedImage.size <= 0:
                raise ValueError("Empty Image")
            if extractedImage.shape[0] != imageSize:
                raise ValueError("Formatting error: size don't match")
            if extractedImage.shape[0] != extractedImage.shape[1]:
                raise ValueError("Image not a square image")

            image = SEMImage(dwellTime, imageSize, extractedImage)
            imageSet.append(image)

        except FileNotFoundError:
            print(f"Error: File not found - {path}")
        except Exception as e:
            print(f"Error processing file {path}: {e}")
    return imageSet


def read_sims_images(SIMSFolderPath):
    imageSet = []
    for eachFolder in os.listdir(SIMSFolderPath):
        sampleFolder = os.path.join(SIMSFolderPath, eachFolder)
        spectrometryImages = []
        imageSize = 0
        dwellTime = 0.0
        totalImage = 0

        if os.path.isdir(sampleFolder):
            tiffFiles = glob.glob(os.path.join(sampleFolder, '*.tiff'))
            csvFile = glob.glob(os.path.join(sampleFolder, '*.csv'))

            for eachImagePath in tiffFiles:
                imageOfOneMass = np.array(Image.open(eachImagePath))
                imageSize = len(imageOfOneMass)
                spectrometryImages.append(imageOfOneMass)

            for eachImagePath in csvFile:
                imageDataOfOneMass = pd.read_csv(eachImagePath, delimiter=';', dtype=str, encoding='ISO-8859-1')
                dwellTime = float(imageDataOfOneMass.iloc[1, 0])
                imageSize = int(imageDataOfOneMass.iloc[1, 1])
                totalImage = np.asarray(imageDataOfOneMass.iloc[21:, :imageSize].astype(float)).astype(np.uint16)

        if spectrometryImages:
            image = SIMSImage(imageSize, dwellTime, spectrometryImages, totalImage)
            imageSet.append(image)

    return imageSet


"""
Execution
import matplotlib.pyplot as plt

path = "C:/Users/akbh02/JupyterNotebooks"
newImageSet = read_sem_images(path)

for image in newImageSet[:2]:
    plt.figure()
    plt.imshow(image.extractedImage, cmap='gray')
    plt.show()
    print('Dwell time(us): ', image.dwellTime)
    print('Image size: ', image.imageSize)
    
"""
