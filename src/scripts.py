import numpy as np
import matplotlib.pyplot as plt
from initialize_database import read_sims_images, read_sem_images
from display_and_analysis_functions import display_mask, display_stitched_image, calculate_psnr
from sparse_image_gen import generate_sparse_image_sims, generate_sparse_image_sem
from random_sampled_image_gen import nn_interpolation_for_sparse_image, generate_random_sparse_image_sem
from stitch_images import stitch_images_sims, stitch_images_sem

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

display_mask(SparseSEMImage, exampleSEMFirst)
display_mask(SparseSIMSImage, exampleSIMSFirst)

display_stitched_image(exampleSEMFirst, exampleSEMSecond, 15, [50, 100, 200, 300])
display_stitched_image(exampleSIMSFirst, exampleSIMSSecond, 15, [50, 100, 200, 300])

stitchedSIMS = stitch_images_sims(exampleSIMSFirst, exampleSIMSSecond, 15, [50, 100, 200, 300])
print(stitchedSIMS.dwellTime)
print(stitchedSIMS.extractedImage)
print(stitchedSIMS.spectrometryImages)

'''