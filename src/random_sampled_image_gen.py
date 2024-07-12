import numpy as np
from scipy.spatial import Voronoi, cKDTree


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


def generate_random_sparse_image_sims(imageObject, sparsityPercent):
    imagesToSample = imageObject.spectrometryImages
    imageSize = imageObject.imageSize
    randomPixelIndices = generate_random_pixel_locations(imageSize, sparsityPercent)
    randomPixelIntensities = np.zeros((len(imagesToSample), len(randomPixelIndices)))
    for i, eachChannelImage in enumerate(imagesToSample):
        imageOpened = np.ravel(eachChannelImage)
        randomPixelIntensities[i] = imageOpened[randomPixelIndices]

    randomSparseFeatures = np.array([randomPixelIndices % imageSize, randomPixelIndices // imageSize,
                                     randomPixelIntensities]).astype(int)

    return RandomSparseImage(randomSparseFeatures, imageSize)


def add_corners_to_sample_set(imageToSample):
    xCornerCoords = np.array([0, 0, len(imageToSample) - 1, len(imageToSample) - 1])
    yCornerCoords = np.array([0, len(imageToSample) - 1, 0, len(imageToSample) - 1])
    cornerPixelIntensities = np.array(
        [imageToSample[xCornerCoords[i], yCornerCoords[i]] for i in range(len(xCornerCoords))])
    return np.array([xCornerCoords, yCornerCoords, cornerPixelIntensities])


def generate_random_pixel_locations(imageSize, sparsityPercent):
    sampleSize = int(imageSize * imageSize * sparsityPercent / 100)
    randomPixelIndices = np.random.choice(imageSize, size=sampleSize, replace=False)
    return randomPixelIndices


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


'''
Execution

path = "D:/Akarsh/Adaptive Scanning/Data/SEM_images_29_May_2024"
availableSEImages = read_sem_images(path)
imageOne = availableSEImages[3]
randomSparseImageTwo = generate_random_sparse_image(imageOne, 50)
interpolatedImage = nn_interpolation_for_sparse_image(randomSparseImageTwo, 3)
plt.figure()
plt.imshow(interpolatedImage, cmap = 'grey')
plt.show()
'''
