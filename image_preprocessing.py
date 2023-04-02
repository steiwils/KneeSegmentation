import numpy as np
import os
from skimage.io import imread, imshow, concatenate_images, imread_collection, show
from itertools import chain

import os 

img_dir = './data/Knee_Segmentation_Dataset/Train_X/*.png' # This is the 'load_pattern' parameter needed for the imread_collection function
ground_dir = './data/Knee_Segmentation_Dataset/Train_Y/*.png' # This is the 'load_pattern' parameter needed for the imread_collection function

# Training images
ground_image_set = imread_collection(ground_dir)
ground_image_list = list(ground_image_set)

train_image_set = imread_collection(img_dir)
train_image_list = list(train_image_set)


# Saving the image in list format for further processing in training phase

np.savez('./data/data_train_list_apt.npz', train_image_list = train_image_list, ground_image_list= ground_image_list)
## Save several arrays into a single file in uncompressed .npz format.

# Show some of the images

imshow(train_image_set[1])
show()
imshow(ground_image_set[1])
show()