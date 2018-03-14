# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 21:40:26 2018
@author: Nazneen Kotwal
"""
import os
import cv2
import math
import csv
import datetime 

os.mkdir('Training')
os.mkdir('Testing')
print(datetime.datetime.now().time())

list1 = []
with open('myfile.csv','r') as csvfile:
     filereader = csv.reader(csvfile, delimiter=',', quotechar='|')
     for row in filereader:
         list1.append(row)
train_image_path = 'C:\\Users\\nazne\\OneDrive\\Documents\\ECE 763 Computer Vision\\Project\\Project1\\Project_try2\\images'
path_test = 'C:\\Users\\nazne\\OneDrive\\Documents\\ECE 763 Computer Vision\\Project\\Project1\\Project_try2\\Testing'
path_train = 'C:\\Users\\nazne\\OneDrive\\Documents\\ECE 763 Computer Vision\\Project\\Project1\\Project_try2\\Training_Images'

num = 1 
for i in range(0,1500):
     full_image = cv2.imread(os.path.join(train_image_path, list1[i][0]))
     x1 = math.floor(float(list1[i][1]))
     y1 = math.floor(float(list1[i][2]))
     x2 = math.floor(float(list1[i][3]))
     y2 = math.floor(float(list1[i][4]))
     crop_image = full_image[y1:y2, x1:x2]
     non_face = full_image[0:60,0:60]
     height, width = crop_image.shape[:2]
     x = 60/width
     y = 60/height
     img_fn = cv2.resize(crop_image,None,fx=x, fy=y, interpolation = cv2.INTER_LINEAR)
     cv2.imwrite(os.path.join(path_train , 'Face_'+str(num)+'.jpg'), img_fn)
     cv2.imwrite(os.path.join(path_train , 'Non_Face_'+str(num)+'.jpg'), non_face)
     num +=1
 
print('Preprocessing done...')
 
num2 = 1
for i in range(1500,1999):
     full_image = cv2.imread(os.path.join(train_image_path, list1[i][0]))
     x1 = math.floor(float(list1[i][1]))
     y1 = math.floor(float(list1[i][2]))
     x2 = math.floor(float(list1[i][3]))
     y2 = math.floor(float(list1[i][4]))
     crop_image = full_image[y1:y2, x1:x2]
     non_face = full_image[0:60,0:60]
     height, width = crop_image.shape[:2]
     x = 60/width
     y = 60/height
     img_fn = cv2.resize(crop_image,None,fx=x, fy=y, interpolation = cv2.INTER_LINEAR)
     cv2.imwrite(os.path.join(path_test , 'Face_'+str(num2)+'.jpg'), img_fn)
     cv2.imwrite(os.path.join(path_test , 'Non_Face_'+str(num2)+'.jpg'), non_face)
     num2 +=1
