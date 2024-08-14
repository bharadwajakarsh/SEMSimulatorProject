import numpy as np
import matplotlib.pyplot as plt
from initialize_database import read_sims_images, read_sem_images
from display_and_analysis_functions import calculate_psnr
from sparse_image_gen import generate_sparse_image_sims, generate_sparse_image_sem
from stitch_images import stitch_images_sims, stitch_images_sem
from read_raw_file import (read_data, get_image_size, process_data, create_channel_range_count_image,
                           plot_channel_range_count_image, plot_total_count_image)

'''
### Adaptive sampling and stitching for SEM and SIMS images

simsPath = 'D:/Akarsh/Adaptive Scanning/Data/SIMS Images/Sample4'
semPath = 'D:/Akarsh/Adaptive Scanning/Data/SEM Images/SEM_images_29_May_2024'

sparsityPercents = np.arange(5, 95, 5)
peakSNRSEM = []
peakSNRSIMS = []

SIMSImageSet = read_sims_images(simsPath)
SEMImageSet = read_sem_images(semPath)

exampleSIMSFirst = SIMSImageSet[3]
exampleSIMSSecond = SIMSImageSet[0]

exampleSEMFirst = SEMImageSet[3]
exampleSEMSecond = SEMImageSet[8]

SparseSIMSImage = generate_sparse_image_sims(exampleSIMSFirst, 15, [50, 100, 200, 300])
SparseSEMImage = generate_sparse_image_sem(exampleSEMFirst, 15, [50, 100, 200, 300])

for sp in sparsityPercents:
    hybridImageSEM = stitch_images_sem(exampleSEMFirst, exampleSEMSecond, sp, [50, 100, 200, 300])
    hybridImageSIMS = stitch_images_sims(exampleSIMSFirst, exampleSIMSSecond, sp, [50, 100, 200, 300])

    peakSNRSEM = np.append(peakSNRSEM, calculate_psnr(exampleSEMSecond.extractedImage, hybridImageSEM.extractedImage))
    peakSNRSIMS = np.append(peakSNRSIMS,
                            calculate_psnr(exampleSIMSSecond.extractedImage, hybridImageSIMS.extractedImage))

plt.figure()
plt.plot(sparsityPercents, peakSNRSEM)
plt.xlabel('sparsity%')
plt.xlabel('sparsity%')
plt.plot(sparsityPercents, peakSNRSIMS)
plt.show()

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
