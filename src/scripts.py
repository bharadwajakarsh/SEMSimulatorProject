from initialize_database import read_sims_images
from display_and_analysis_functions import display_mask
from sparse_image_gen import generate_sparse_image_sims

path = 'D:/Akarsh/Adaptive Scanning/Data/24_May_2024/SIMSImages'
SIMSImageSet = read_sims_images(path)
exampleSIMS = SIMSImageSet[2]
SparseSIMSImage = generate_sparse_image_sims(exampleSIMS, 15, [50, 100, 200, 300])
display_mask(SparseSIMSImage, exampleSIMS)