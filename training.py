import numpy as np
import matplotlib
import os
import matplotlib.pyplot as plt
import cv2 # Computer Vision library
from tqdm import tqdm_notebook, tnrange
import skimage # sci-kit learn image manipulation package
from itertools import chain # Can be used to combine two iterables into one
from skimage.io import imread, imshow, concatenate_images, imread_collection
from skimage.transform import resize # Allows us to resize pixel size of our images
from skimage.morphology import label # https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.label
from sklearn.model_selection import train_test_split # Training and validation set splitting tool
import random # used to generate pseudo-random numbers
from numpy import load


import tensorflow as tf
from keras import backend as K
from keras.backend import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
#from keras import backend as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.python.client import device_lib


#print(device_lib.list_local_devices())

# Setting device to GPU
with tf.device('/device:GPU:0'):

    im_width = 224
    im_height = 224


## Load training images
    data1 = load('./data/data_train_list_apt.npz')
    train_image_list = data1['train_image_list']


    # Building the X training set
    X_list = []
    for img in train_image_list:
        img = img / 255 # Normalizing pixel values in grayscale
        X_list.append(img)

    X_array = np.asarray(X_list) # Shape: (84, 224, 244)
    X_array = np.reshape(X_array, (X_array.shape[0], X_array.shape[1], X_array.shape[2], 1)) # Shape after: (84, 224, 224, 1)
    # Reshaped to add the "channel" to these images. Only one channel since we are in grayscale.

## Loading ground/ annotated images list
    data = load('./data/data_train_list_apt.npz', allow_pickle = True)
    ground_image_list = data['ground_image_list']

## Checking the size of all the ground images
 # making a dictionary of all the different image sizes.
    dic = {}
    for img in ground_image_list:
        if img.shape in dic:
            dic[img.shape] += 1
        else:
            dic[img.shape] = 1

    #print(dic)

## Making ground image training set

    Y_list = []

    for img in ground_image_list:
        img = img / 255 # Normalizing pizel values in grayscale
        Y_list.append(img)

    Y_array = np.asarray(Y_list)
    Y_array = np.reshape(Y_array, (Y_array.shape[0], Y_array.shape[1], Y_array.shape[2], 1))



