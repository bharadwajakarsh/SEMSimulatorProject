from initialize_database import read_sims_images, read_sem_images
from display_and_analysis_functions import display_mask
from sparse_image_gen import generate_sparse_image_sims, generate_sparse_image_sem

simsPath = 'D:/Akarsh/Adaptive Scanning/Data/24_May_2024/SIMSImages'
semPath = 'D:/Akarsh/Adaptive Scanning/Data/SEM_images_29_May_2024'

SIMSImageSet = read_sims_images(simsPath)
SEMImageSet = read_sem_images(semPath)

exampleSIMS = SIMSImageSet[3]
exampleSEM = SEMImageSet[3]

SparseSIMSImage = generate_sparse_image_sims(exampleSIMS, 15, [50, 100, 200, 300])
SparseSEMImage = generate_sparse_image_sem(exampleSEM, 15, [50, 100, 200, 300])

display_mask(SparseSEMImage, exampleSEM)
display_mask(SparseSIMSImage, exampleSIMS)