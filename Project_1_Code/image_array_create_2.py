# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 21:54:38 2018

@author: Nazneen Kotwal
"""

import numpy as np
import os
import cv2
import datetime 

train_image_path = 'C:\\Users\\nazne\\OneDrive\\Documents\\ECE 763 Computer Vision\\Project\\Project1\\Project_try2\\images'
path_test = 'C:\\Users\\nazne\\OneDrive\\Documents\\ECE 763 Computer Vision\\Project\\Project1\\Project_try2\\Testing'
path_train = 'C:\\Users\\nazne\\OneDrive\\Documents\\ECE 763 Computer Vision\\Project\\Project1\\Project_try2\\Training_Images'

def image_array_create(no_of_train,no_of_test,image_size,img_type):
    print(datetime.datetime.now().time())
    a= 1
    b = 1
    if (img_type is 'GRAY'):
        t = np.zeros([no_of_train,image_size*image_size])
        u = np.zeros([no_of_train,image_size*image_size])
        t1 = np.zeros([no_of_test,image_size*image_size])
        u1 = np.zeros([no_of_test,image_size*image_size])
    else:
        t = np.zeros([no_of_train,image_size*image_size*3])
        u = np.zeros([no_of_train,image_size*image_size*3])
        t1 = np.zeros([no_of_test,image_size*image_size*3])
        u1 = np.zeros([no_of_test,image_size*image_size*3])
        
    for i in range(0,no_of_train):
        face = cv2.imread((os.path.join(path_train , 'Face_'+str(a)+'.jpg')),1)
        nonface = cv2.imread((os.path.join(path_train , 'Non_Face_'+str(a)+'.jpg')),1)
        if (img_type is 'GRAY'):
            face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
            nonface = cv2.cvtColor(nonface, cv2.COLOR_RGB2GRAY)
            face = cv2.resize(face,(image_size,image_size))
            nonface = cv2.resize(nonface,(image_size,image_size))
        t[i,:] = np.reshape(face, (1,-1))
        u[i,:] = np.reshape(nonface, (1,-1))
#        t[i,:] = face.flatten()
#        u[i,:] = nonface.flatten()
        a +=1
        
    for i in range(0,no_of_test):
        face = cv2.imread((os.path.join(path_test , 'Face_'+str(b)+'.jpg')),1)
        nonface = cv2.imread((os.path.join(path_test , 'Non_Face_'+str(b)+'.jpg')),1)
        if (img_type is 'GRAY'):
            face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
            nonface = cv2.cvtColor(nonface, cv2.COLOR_RGB2GRAY)
            face = cv2.resize(face,(image_size,image_size))
            nonface = cv2.resize(nonface,(image_size,image_size))
#        t1[i,:] = np.reshape(face, (1,-1))
#        u1[i,:] = np.reshape(nonface, (1,-1))
        t1[i,:] = face.flatten()
        u1[i,:] = nonface.flatten()
        b +=1
        
    return(t1,u1,t,u)