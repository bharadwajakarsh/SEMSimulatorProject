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


class RandomSparseImage:
    def __init__(self, randomSparseFeatures, imageSize):
        self.randomSparseFeatures = randomSparseFeatures
        self.imageSize = imageSize


class SparseImageSIMS:
    def __init__(self, sparseFeatures, imageSize):
        self.sparseFeatures = sparseFeatures
        self.imageSize = imageSize


class SparseImageSEM:
    def __init__(self, sparseFeatures, imageSize):
        self.sparseFeatures = sparseFeatures
        self.imageSize = imageSize
