import numpy as np

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
