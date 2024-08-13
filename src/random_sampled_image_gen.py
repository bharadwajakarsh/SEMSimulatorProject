import numpy as np
import multiprocessing as mp
from scipy.spatial import Voronoi, cKDTree
from pykrige.ok import OrdinaryKriging


class RandomSparseImage:
    def __init__(self, randomSparseFeatures, imageSize):
        self.randomSparseFeatures = randomSparseFeatures
        self.imageSize = imageSize


def generate_random_sparse_image_sem(imageObject, sparsityPercent):
    imageToSample = imageObject.extractedImage
    imageSize = imageObject.imageSize
    randomPixelIndices = generate_random_pixel_locations(imageSize, sparsityPercent)
    imageOpened = np.ravel(imageToSample)
    randomPixelIntensities = imageOpened[randomPixelIndices]
    randomSparseFeatures = np.array([randomPixelIndices % imageSize, randomPixelIndices // imageSize,
                                     randomPixelIntensities]).astype(int)

    cornersToAdd = add_corners_to_sample_set(imageToSample)

    for i in range(len(cornersToAdd)):
        if not np.any(np.all(randomSparseFeatures == cornersToAdd[:, i][:, None], axis=0)):
            randomSparseFeatures = np.concatenate((randomSparseFeatures, cornersToAdd[:, i][:, None]), axis=1)

    return RandomSparseImage(randomSparseFeatures, imageSize)


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


def interpolate_pixel(x, y, points, vorObject, kDTree, numberNeighbours, randomPixelIntensities):
    distances, closestPointToPixel = kDTree.query([x, y], k=numberNeighbours)

    if len(closestPointToPixel) < numberNeighbours:
        return np.mean(randomPixelIntensities[closestPointToPixel])

    regionIndex = vorObject.point_region[closestPointToPixel[0]]
    region = vorObject.regions[regionIndex]

    if -1 in region:
        return 0

    vertices = vorObject.vertices[region]
    neighbourValues = []

    for eachVertex in vertices:
        distances = np.linalg.norm(points - eachVertex, axis=1)
        closestPointIndex = np.argmin(distances)
        neighbourValues.append(randomPixelIntensities[closestPointIndex])

    weights = np.array(
        [np.linalg.norm(np.cross(vertices[(i + 1) % len(vertices)] - [x, y], vertices[i] - [x, y])) for i in
         range(len(vertices))])
    weights /= weights.sum()

    return np.dot(weights, neighbourValues)


def process_pixel(args):
    x, y, points, vorObject, kDTree, numberNeighbours, randomPixelIntensities = args
    return x, y, interpolate_pixel(x, y, points, vorObject, kDTree, numberNeighbours, randomPixelIntensities)


def nn_interpolation_for_sparse_image(randomSparseImageObject, numberNeighbours):
    randomSparseFeatures = randomSparseImageObject.randomSparseFeatures
    imageSize = randomSparseImageObject.imageSize

    xCoords, yCoords, randomPixelIntensities = randomSparseFeatures

    interpolatedImage = np.zeros((imageSize, imageSize))
    points = np.column_stack((xCoords, yCoords))

    vorObject = Voronoi(points)
    kDTree = cKDTree(points)

    knownPixels = set(zip(xCoords, yCoords))
    pixelArguments = []

    for i in range(imageSize):
        for j in range(imageSize):
            if (j, i) not in knownPixels:
                pixelArguments.append((j, i, points, vorObject, kDTree, numberNeighbours, randomPixelIntensities))
            else:
                knownPixelIndices = np.where((xCoords == j) & (yCoords == i))[0]
                interpolatedImage[i, j] = randomPixelIntensities[knownPixelIndices[0]]

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(process_pixel, pixelArguments)

    for i, j, pixelValue in results:
        interpolatedImage[i, j] = pixelValue

    return interpolatedImage


def kriging_interpolation(randomSparseImageObject):
    randomSparseFeatures = randomSparseImageObject.randomSparseFeatures
    imageSize = randomSparseImageObject.imageSize

    xCoords = np.array(randomSparseFeatures[0]).astype(np.int64)
    yCoords = np.array(randomSparseFeatures[1]).astype(np.int64)
    randomPixelIntensities = np.array(randomSparseFeatures[2]).astype(np.float64)

    gridX = np.arange(0, imageSize, dtype=np.float64)
    gridY = np.arange(0, imageSize, dtype=np.float64)

    OK = OrdinaryKriging(xCoords, yCoords, randomPixelIntensities, variogram_model='linear', verbose=False,
                         enable_plotting=False)
    interpolatedValues, ss = OK.execute('grid', gridX, gridY)

    return interpolatedValues
