import numpy as np
import matplotlib.pyplot as plt
from read_images import read_sims_images, read_sem_images
from display_and_analysis_functions import calculate_psnr, calculate_ssim
from sparse_image_gen import generate_sparse_image_sims, generate_sparse_image_sem, extract_sparse_features_sem
from stitch_images import stitch_images_sims, stitch_images_sem
from read_raw_file import (read_data, get_image_size, process_data, create_channel_range_count_image,
                           plot_channel_range_count_image, plot_total_count_image)

### Compare with emperical results

semPath = 'D:/Akarsh/Adaptive Scanning/Data/16August_first_Adaptive_Scan_Test/Experiment 3'
SEMImageSet = read_sem_images(semPath)
sampleImages = SEMImageSet[0:2]
testImage = SEMImageSet[2]
highDTImage = SEMImageSet[3]

stitchedImage = stitch_images_sem(testImage, [sampleImages[0]], 20)
yCoords, xCoords, dc0, dc1 = extract_sparse_features_sem(testImage.extractedImage, 20, [200])
yCoords = yCoords.astype(int)
xCoords = xCoords.astype(int)
empStitchedImage = np.copy(testImage.extractedImage)
empStitchedImage[yCoords, xCoords] = highDTImage.extractedImage[yCoords, xCoords]

plt.figure()
plt.title(f"Low DT Image, {testImage.dwellTime}")
plt.imshow(testImage.extractedImage, cmap='grey')
plt.show()

plt.figure()
plt.title(f"Simulated stitched image, {stitchedImage.dwellTime}")
plt.imshow(stitchedImage.extractedImage, cmap='grey')
plt.show()

plt.figure()
plt.title(f"Empirical stitched image")
plt.imshow(empStitchedImage, cmap='grey')
plt.show()
print(f"SSIM for original image and 200us image {calculate_ssim(testImage.extractedImage, sampleImages[1].extractedImage)}")
print(f"SSIM for emp. stitched image and 200us image {calculate_ssim(empStitchedImage, sampleImages[1].extractedImage)}")
print(f"SSIM for sim. stitched image and 200us image {calculate_ssim(stitchedImage.extractedImage, sampleImages[1].extractedImage)}")

'''

### Adaptive sampling and stitching for SEM and SIMS images

simsPath = 'D:/Akarsh/Adaptive Scanning/Data/SIMS Images/Sample4'
semPath = 'D:/Akarsh/Adaptive Scanning/Data/SEM Images/SEM_images_29_May_2024'

sparsityPercents = np.arange(5, 95, 5)
peakSNRSEM = []
peakSNRSIMS = []

SIMSImageSet = read_sims_images(simsPath)
SEMImageSet = read_sem_images(semPath)

exampleSIMSFirst = SIMSImageSet[-1]
exampleSIMSSeconds = SIMSImageSet[:-1]

exampleSEMFirst = SEMImageSet[3]
exampleSEMSeconds = SEMImageSet[4:8]

SparseSIMSImage = generate_sparse_image_sims(exampleSIMSFirst, 15, [each.dwellTime for each in exampleSIMSSeconds])
SparseSEMImage = generate_sparse_image_sem(exampleSEMFirst, 15, [each.dwellTime for each in exampleSEMSeconds])

for sp in sparsityPercents:
    hybridImageSEM = stitch_images_sem(exampleSEMFirst, exampleSEMSeconds, sp)
    hybridImageSIMS = stitch_images_sims(exampleSIMSFirst, exampleSIMSSeconds, sp)

    peakSNRSEM = np.append(peakSNRSEM,
                           calculate_psnr(exampleSEMSeconds[-1].extractedImage, hybridImageSEM.extractedImage))
    peakSNRSIMS = np.append(peakSNRSIMS,
                            calculate_psnr(exampleSIMSSeconds[-1].extractedImage, hybridImageSIMS.extractedImage))

plt.figure()
plt.plot(sparsityPercents, peakSNRSEM)
plt.xlabel('sparsity%')
plt.xlabel('sparsity%')
plt.plot(sparsityPercents, peakSNRSIMS)
plt.show()

'''

'''
### Read raw file and generating "mass-images"

fileName = 'D:/Akarsh/Adaptive Scanning/Data/SIMS Images/Sample3/testnormal.raw'
channelLowerBound = 5000
channelUpperBound = 6000
channel = 6247

rawData = read_data(fileName)
imageSize = get_image_size(rawData)
totalCountImage, channel_count_image = process_data(fileName, channel)

rangeCountImage = create_channel_range_count_image(rawData, imageSize, channelLowerBound, channelUpperBound)

print("Total Count Image:\n", totalCountImage)
print(f"Channel {channel} Count Image:\n", channel_count_image)

plot_channel_range_count_image(rangeCountImage, channelLowerBound, channelUpperBound)

'''
