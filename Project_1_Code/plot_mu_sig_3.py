# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 22:20:36 2018

@author: Nazneen Kotwla
"""

import cv2
import numpy as np

def plot_mu_sig (mu_1,sig_1,title1,title2,img_type,image_size,ploting):
    # Plot Mean  
    mu = mu_1 / np.max(mu_1)
    if (img_type is 'GRAY'):
        mu_mat = np.reshape(mu,(image_size,image_size))
    else:
        mu_mat = np.reshape(mu,(image_size,image_size,3))
    r = 200.0 / mu_mat.shape[1]
    dim = (200, int(mu_mat.shape[0] * r))
    resized = cv2.resize(mu_mat, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("resized", resized)
    resized = resized*(255/np.max(resized))
    k = cv2.waitKey(0) & 0xFF
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('C:/Users/nazne/OneDrive/Documents/ECE 763 Computer Vision/Project/Project1/Project_try2/Results/'+title1+'.png',resized.astype('uint8'))
        cv2.destroyAllWindows()
     
    if (ploting is 'BOTH'):
        #% Plot covariane
        sig = sig_1 / np.max(sig_1)
        if (img_type is 'GRAY'):
            sig_mat = np.reshape((sig),(image_size,image_size))
        else:
            sig_mat = np.reshape((sig),(image_size,image_size,3))
        r = 200.0 / sig_mat.shape[1]
        dim = (200, int(sig_mat.shape[0] * r))
        # perform the actual resizing of the image and show it
        resized = cv2.resize(sig_mat, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow("resized", resized)
        resized = resized*(255/np.max(resized))
        k = cv2.waitKey(0) & 0xFF
        if k == 27:         # wait for ESC key to exit
            cv2.destroyAllWindows()
        elif k == ord('s'): # wait for 's' key to save and exit
            cv2.imwrite('C:/Users/nazne/OneDrive/Documents/ECE 763 Computer Vision/Project/Project1/Project_try2/Results/'+title2+'.png',resized)
            cv2.destroyAllWindows()
             