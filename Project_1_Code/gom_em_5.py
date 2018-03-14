# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 19:49:05 2018
@author: Nazneen Kotwal
"""
## Mixture of Gaussian
import numpy as np
from scipy.stats import multivariate_normal

def MOG_EM(Xin , K):
    np.random.seed(0)
    iterations = 0
    Xin = np.array(Xin)
    # Initialising the parameters
    (rows,col) = np.shape(Xin)
    previous_L = 10000
    L = 0
    sig = [[]] * K
    lam = np.matlib.repmat(1/K, K, 1)
    K_integers = np.random.permutation(rows);
    K_integers = np.array(K_integers[0:K])
    means = np.zeros([K,col])
    for i in range(len(K_integers)):
        means[i,:] = Xin[K_integers[i],:] 
    
    dataset_mean = np.divide(np.sum(Xin,axis = 0), rows)
    variance = np.zeros([col, col])

    for i in range(rows):
        mat = Xin[i,:] - dataset_mean;
        mat = np.transpose(mat) * mat
        variance = variance + mat;
#        variance = np.cov(Xin, rowvar=False, bias=1, ddof=None)   
    variance = np.diagonal(variance)  
    variance = np.diag(variance,0) + np.exp(-1*6)
    
    for i in range (K):
        sig[i] = variance;
    
    epsolon = 0.01
    l = np.zeros([rows,K])
    l1 = np.zeros([rows,K])
    r = np.zeros([rows,K])
    while True :
        # Estep 
        for k in range(K):
            l1[:,k] = multivariate_normal.pdf(Xin, mean=means[k,:], cov=sig[k])
            a = lam[k]
            l[:,k]=(a * l1[:,k])
        s = np.sum(l,axis = 1)     
#        for k in range(K):
#            l1[:,k] = multi_norm_pdf_file.multi_norm_pdf(Xin,means[k,:], sig[k])
#            a = lam[k]
#            l[:,k]=(a * l1[:,k])
#        s = np.sum(l,axis = 1) 
        
        for i in range(rows):
            r[i,:] = np.divide(l[i,:],s[i])
                
        # M Step
        r_summed_rows = np.sum(r,axis=0) 
        r_summed_all = np.sum(np.sum(r,axis=0));
        for k in range(K):
            lam[k] = r_summed_rows[k] / r_summed_all;
            
            new_mu = np.zeros([1,col]);
            for i in range(rows):
                new_mu = new_mu + (r[i,k]*Xin[i,:]);
            means[k,:] = np.divide(new_mu,r_summed_rows[k]);
                
            new_sigma = np.zeros([col,col]);
            for i in range(rows):
                mat = Xin[i,:] - means[k,:]
                mat = r[i,k] * (np.transpose(mat) * mat)
                new_sigma = new_sigma + mat;
            sig[k] = np.divide(new_sigma,r_summed_rows[k]);
            sig[k] = np.diagonal(sig[k])  
            sig[k] = np.diag(sig[k],0) + np.exp(-1*6)
            
        temp = np.zeros([rows,K]);
        for k in range(K):
            l2 = multivariate_normal.pdf(Xin, means[k,:], sig[k],allow_singular=True)
            a = lam[k]
            temp[:,k] = a * l2;         
        temp = np.sum(temp,axis=1);
        temp = np.log(temp);        
        L = np.sum(temp,axis=0);  
        
        iterations = iterations + 1;        
        print(str(iterations)+' : '+str(L))
        if (abs(L - previous_L) < epsolon) or (iterations == 1000):
            break
        previous_L = L
    return (lam,means,sig)
            