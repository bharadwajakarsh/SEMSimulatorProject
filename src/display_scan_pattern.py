import numpy as np
import matplotlib.pyplot as plt

from src.generate_new_images import generate_new_images
from src.intialize_database import refresh_database
from src.sparse_image_gen import generate_sparse_image


def display_scan_pattern(lowDTimageObject, sparsityPercent, maxDwellTime, minDwellTime):
    sparseImageObject = generate_sparse_image(lowDTimageObject, sparsityPercent, maxDwellTime, minDwellTime)
    imgSize = sparseImageObject.imageSize

    impPixelCoords = sparseImageObject.sparseFeatures[0, :].astype(int)
    sortedIntensities = np.argsort(sparseImageObject.sparseFeatures[1, :])[::-1]

    sortedPixelCoords = impPixelCoords[sortedIntensities]

    xcoords = sortedPixelCoords // imgSize
    ycoords = sortedPixelCoords % imgSize

    plt.figure(figsize=(20, 20))
    plt.imshow(lowDTimageObject.extractedImage, cmap='grey')
    plt.plot(xcoords[:1000], ycoords[:1000], color='white', linewidth=1)
    plt.show()


"""
Execution

path = "C:/Users/akbh02/JupyterNotebooks"
availableImages = refresh_database(path)
testImage = generate_new_images(availableImages, 10, 10)[5]
display_scan_pattern(testImage, 15, 300, 10)

"""