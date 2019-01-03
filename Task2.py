# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 19:56:09 2018

@author: Ayush
"""

import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
#from kt_utils import *
import pandas as pd

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

#matplotlib inline

noise = pd.read_csv('noise_classified.csv',header=None)
labels_noise= pd.read_csv("attribute_list.csv",skiprows=1)
labels= labels_noise[labels_noise['noise']==1]
train_test_data_smiling=labels.loc[:,['file_name','smiling']]