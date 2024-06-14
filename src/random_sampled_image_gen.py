import numpy as np
from scipy.spatial import Voronoi, cKDTree


class RandomSparseImage:
    def __init__(self, randomSparseFeatures, imageSize):
        self.randomSparseFeatures = randomSparseFeatures
        self.imageSize = imageSize


def generate_random_sparse_image(imageObject, sparsityPercent):
    randomSparseFeatures = randomly_sample_image(imageObject, sparsityPercent)
    imageSize = imageObject.imageSize
    return RandomSparseImage(randomSparseFeatures, imageSize)


def randomly_sample_image(imageObject, sparsityPercent):
    imageToSample = imageObject.extractedImage
    sampleSize = int(imageToSample.shape[0] * imageToSample.shape[1] * sparsityPercent / 100)
    imageOpened = np.ravel(imageToSample)
    randomPixelIndices = np.random.choice(imageToSample.size, size=sampleSize, replace=False)
    randomPixelIntensities = imageOpened[randomPixelIndices]
    return np.array([randomPixelIndices % imageToSample.shape[0], randomPixelIndices // imageToSample.shape[0],
                     randomPixelIntensities]).astype(int)


def interpolate_pixel(x, y, points, vorObject, kDTree, numberNeighbours, randomPixelIntensities):
    distances, closestPointToPixel = kDTree.query([x, y], k=numberNeighbours)

    if len(closestPointToPixel) < numberNeighbours:
        return np.mean(randomPixelIntensities[closestPointToPixel])

    regionIndex = vorObject.point_region[closestPointToPixel[0]]
    region = vorObject.regions[regionIndex]

    if -1 in region:
        return np.nan

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


def nn_interpolation_for_sparse_image(randomSparseImageObject, numberNeighbours):
    randomSparseFeatures = randomSparseImageObject.randomSparseFeatures
    imageSize = randomSparseImageObject.imageSize

    xCoords = randomSparseFeatures[0]
    yCoords = randomSparseFeatures[1]
    randomPixelIntensities = randomSparseFeatures[2]

    interpolatedImage = np.zeros((imageSize, imageSize))
    points = np.column_stack((xCoords, yCoords))

    vorObject = Voronoi(points)
    kDTree = cKDTree(points)

    knownPixels = set(zip(xCoords, yCoords))
    count = 0

    for i in range(imageSize):
        for j in range(imageSize):
            if (j, i) not in knownPixels:
                interpolatedImage[i, j] = interpolate_pixel(j, i, points, vorObject, kDTree, numberNeighbours,
                                                            randomPixelIntensities)
                count = count + 1
                print(count)
            else:
                knownPixelIndices = np.where((xCoords == j) & (yCoords == i))[0]
                interpolatedImage[i, j] = randomPixelIntensities[knownPixelIndices[0]]

    return interpolatedImage
