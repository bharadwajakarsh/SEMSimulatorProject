from initialize_database import read_sims_images, read_sem_images
from display_and_analysis_functions import display_mask, display_stitched_image
from sparse_image_gen import generate_sparse_image_sims, generate_sparse_image_sem
from stitch_images import stitch_images_sims

simsPath = 'D:/Akarsh/Adaptive Scanning/Data/24_May_2024/SIMSImages'
semPath = 'D:/Akarsh/Adaptive Scanning/Data/SEM_images_29_May_2024'

SIMSImageSet = read_sims_images(simsPath)
SEMImageSet = read_sem_images(semPath)

exampleSIMSFirst = SIMSImageSet[4]
exampleSIMSSecond = SIMSImageSet[0]

exampleSEMFirst = SEMImageSet[3]
exampleSEMSecond = SEMImageSet[8]

SparseSIMSImage = generate_sparse_image_sims(exampleSIMSFirst, 15, [50, 100, 200, 300])
SparseSEMImage = generate_sparse_image_sem(exampleSEMFirst, 15, [50, 100, 200, 300])

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