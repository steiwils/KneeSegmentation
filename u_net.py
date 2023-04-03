import numpy as np
import matplotlib
import os
import matplotlib.pyplot as plt
import  cv2
from tqdm import tqdm_notebook, tnrange
import skimage
#import scikit-image as skimage
from itertools import chain
from skimage.io import imread, imshow, concatenate_images, imread_collection
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score #, jaccard_similarity_score
import random
import tensorflow as tf

from keras import backend as K
from keras.backend import *
from keras.models import *
from numpy import load
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


from tensorflow.python.client import device_lib

# U-net Architecture:

def conv2d_block(input_tensor, n_filters, kernel_size, batchnorm = True):

    #first layer
    x = Conv2D(filters=n_filters, kernel_size= (kernel_size, kernel_size), kernel_initializer= "he_normal",
               padding= "same")(input_tensor) # We are using the functional API of tensorflow
    
    if batchnorm:
        x = BatchNormalization()(x) # if we are using the batchnorm parameter, add BatchNormalization to the functional sequence
    
    x = Activation("relu")(x)

    #second layer
    x = Conv2D(filters=n_filters, kernel_size= (kernel_size, kernel_size), kernel_initializer= "he_normal",
               padding= "same")(x)
    
    if batchnorm:
        x = BatchNormalization()(x)

    x = Activation("relu")(x)
    
    return x

def get_unet(input_img, n_filters = 16, dropout = 0.5, batchnorm = True):

    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2,2))(c1)
    p1 = Dropout(dropout * 0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2,2))(c2)
    p2 = Dropout(dropout * 0.5)(p2)

    c5 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    # expansion path
    u6 = Conv2DTranspose(n_filters*8, (3,3), strides = (2,2), padding='same')(c5)
    # the 'Conv2DTranspose' can also be thought of as a deconvolution.
    u6 = concatenate([u6, c2]) # adding the earlier corresponding feature map
    u6 = Dropout(dropout)(u7)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3,3), strides=(2,2), padding=3, batchnorm= batchnorm)
    u7 = concatenate([u7, c1])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1,1), activation='sigmoid')(c7)
    model = Model(inputs = [input_img], outputs = [outputs])
    return model