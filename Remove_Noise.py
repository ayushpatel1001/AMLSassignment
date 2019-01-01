
# coding: utf-8

# In[5]:


#creates noise classified.csv by detecting if theres a face in the pictur
import numpy as np
import cv2
import dlib

base_dir='floyd\input\dataset\\'#change to wherever dataset is 
detector = dlib.get_frontal_face_detector()

faces=[]
for x in range (1,5001):# change to number of (files +1)
    img_path=(base_dir + str(x)+'.png')
    #print (img_path)
    img = cv2.imread(img_path)
    
    gray = img.astype('uint8')
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')
    rects = detector(gray, 1)#checks if theres a frontal face
    if len(rects)==0:
        faces.append([x,0])
    else:
        faces.append([x,1])
    

npfaces=np.array(faces)
np.savetxt("noise_classified.csv", npfaces, delimiter=",")#returns a csv with filename and whether or not its noise


# In[6]:




