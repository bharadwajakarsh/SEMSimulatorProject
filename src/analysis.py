import numpy as np
from skimage.metrics import structural_similarity as ssim


def calculate_psnr(originalHDTImage, hybridImage):
    if np.linalg.norm(originalHDTImage - hybridImage) == 0:
        return float('inf')

    return 20 * np.log10(np.max(hybridImage) /
                         np.sqrt(np.mean((originalHDTImage - hybridImage) ** 2)))


def calculate_ssim(stitchedImage, averagedImage):
    if stitchedImage.shape != averagedImage.shape:
        raise ValueError("Images must have the same dimensions")

    ssimValue = ssim(stitchedImage, averagedImage, data_range=255.0)

    return ssimValue
