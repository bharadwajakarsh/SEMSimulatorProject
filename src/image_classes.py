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


class SparseImage:
    def __init__(self, imageType, sparseFeatures, imageSize):
        self.imageType = imageType
        self.sparseFeatures = sparseFeatures
        self.imageSize = imageSize
