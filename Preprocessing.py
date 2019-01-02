
# coding: utf-8

# In[5]:


import numpy as np
from keras.preprocessing import image
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from sklearn.model_selection import train_test_split
import cv2
import dlib
img_dir='floyd\input\dataset\\'
from lab2_landmarks import run_dlib_shape
import pandas as pd


# In[7]:


def preprocessRGB(attribute1,testsize,pictureshape):#testsize=percentage of test data 
    img_dir='floyd\input\dataset\\'
    noise = pd.read_csv('noise_classified.csv',header=None)
    labels_noise= pd.read_csv("attribute_list.csv",skiprows=1)
    labels_noise['noise']=noise.loc[:,1]
    labels= labels_noise[labels_noise['noise']==1]
    attribute=attribute1
    train_test_data=labels.loc[:,['file_name',attribute]]
    train_test_data.loc[:,['file_name',attribute]]
    train_test_data[attribute]= train_test_data[attribute].apply(lambda x: 0 if x < 1  else 1)
    
    
    
    train, test = train_test_split(train_test_data, test_size=testsize)
    y_train= np.array(train[attribute]).T
    y_test=np.array(test[attribute]).T
    train.shape[0]
    picture_shape=pictureshape
    
    i=0
    x_train=np.zeros((len(train[attribute]),picture_shape[0],picture_shape[1],3))
    for x in list(train['file_name']):
        temp= image.load_img(img_dir+str(x)+'.png',target_size=(picture_shape))
        x_train[i,:,:,:]=image.img_to_array(temp)
        i=i+1
    i=0
    x_test=np.zeros((len(test[attribute]),picture_shape[0],picture_shape[1],3))
    for x in list(test['file_name']):
        temp= image.load_img(img_dir+str(x)+'.png',target_size=(picture_shape))
        x_test[i,:,:,:]=image.img_to_array(temp)
        i=i+1
    x_train,x_test = x_train/255,x_test/255
    x_testfiles=test['file_name']
    return x_testfiles,x_train,y_train,x_test,y_test


def facefeaturesdlib(attribute1,testsize,pictureshape):
    noise = pd.read_csv('noise_classified.csv',header=None)
    labels_noise= pd.read_csv("attribute_list.csv",skiprows=1)
    labels_noise['noise']=noise.loc[:,1]
    labels= labels_noise[labels_noise['noise']==1]
    attribute=attribute1
    train_test_data=labels.loc[:,['file_name',attribute]]
    train_test_data.loc[:,['file_name',attribute]]
    train_test_data[attribute]= train_test_data[attribute].apply(lambda x: 0 if x < 1  else 1)
    train, test = train_test_split(train_test_data, test_size=testsize)
    y_train= np.array(train[attribute]).T
    y_test=np.array(test[attribute]).T
    y_train.shape
    
    i=0
    x_train=np.zeros((len(train[attribute]),68,2))
    from keras.preprocessing import image
    for x in list(train['file_name']):
        img= image.load_img(img_dir+str(x)+'.png',target_size=None,interpolation='bicubic')
        img=image.img_to_array(img)
        x_train[i,:,:], _=run_dlib_shape(img)
        i=i+1

    i=0
    x_test=np.zeros((len(test[attribute]),68,2))    
    for x in list(test['file_name']):
        img= image.load_img(img_dir+str(x)+'.png',target_size=None,interpolation='bicubic')
        img=image.img_to_array(img)
        x_test[i,:,:], _=run_dlib_shape(img)
        i=i+1
    x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*2),'f')
    df = pd.DataFrame(x_test)
    df['y']=y_test
    fdf=df.dropna(axis=0)
    t=fdf.shape[0]
    x_test=fdf.iloc[:,:-1]
    y_test=fdf['y']
    x_test=np.array(x_test)
    y_test=np.array(y_test)
    x_test=np.reshape(x_test,(x_test.shape[0],68,2),'f')
    
    x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1]*2),'f')
    df = pd.DataFrame(x_train)
    df['y']=y_train
    fdf=df.dropna(axis=0)
    t=fdf.shape[0]
    x_train=fdf.iloc[:,:-1]
    y_train=fdf['y']
    x_train=np.array(x_train)
    y_train=np.array(y_train)
    x_train=np.reshape(x_train,(x_train.shape[0],68,2),'f')
    
    
    
    return x_train,y_train,x_test,y_test




