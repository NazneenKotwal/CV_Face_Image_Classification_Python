# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 15:01:40 2018

@author: Nazneen Kotwal
"""
"""
  Description: Fitting the t-distribution.
  Input: x - a matrix where each row is one datapoint,
         precision - the algorithm stops when the difference between
                     the previous and the new likelihood is < precision.
                     Typically this is a small number like 0.01.
  Output: mu  - the mean of the fitted t-distribution,
         sig - the scale matrix,
         nu - degrees of freedom.
"""

import numpy as np
import fit_tmix_cost_file
import scipy 
import pdf_tm_file2

def fit_mix_t(x, precision,K):
    print('Performing Mixture of fit_t...')
#    K = 3
#    N = 10;
#    data_mu = [1 ,2]
#    data_sig = [[2, 0], [0, .5]]
#    x = np.random.multivariate_normal(data_mu, data_sig,N);
    np.random.seed(0)
    (I,D) = np.shape(x)
    sig = [[]] * K
    lam = np.matlib.repmat(1/K, K, 1)
    
    K_integers = np.random.permutation(I);
    K_integers = K_integers[0:K]
    means = x[K_integers,:] 
          
    dataset_cov = np.cov(x,rowvar=False, bias=1, ddof=None)
    dataset_cov = np.diagonal(dataset_cov)  
    dataset_variance = np.diag(dataset_cov,0) + (10**-6)
    
    for i in range (K):
        sig[i] = dataset_variance;
        
    ##Initialize degrees of freedom to 100 (just a random large value).
    nu = np.matlib.repmat(10, K, 1)       
    ##The main loop.
    iterations = 0    
    previous_L = 1000000 # just a random initialization
#    count = 0
    while True:
#        count = 1
        #Expectation step.
        #Compute tau
        tau = np.zeros([I,K])
        for k in range(K):
            temp_tau = pdf_tm_file2.pdf_tm(x,means[k,:],sig[k],nu[k])
            tau[:,k] = (lam[k] * temp_tau)
        tau_sum = np.sum(tau,axis = 1)
        tau = tau / np.reshape(tau_sum,(I,-1))
    
        
        delta = np.zeros([I,K])
        for i in range(I):
            for k in range(K):
#                inverse_sig = np.diag(1 / np.diag(sig[k]))
                delta[i,k] = (scipy.spatial.distance.mahalanobis(np.reshape(x[i,:],(1,-1)),means[k],np.linalg.inv(sig[k]))**2)
    
        nu_plus_delta = np.zeros([I,K])
        E_hi = np.zeros([I,K])
        nu_plus_D = nu + D
        nu_plus_delta = np.transpose(nu) + delta
        for i in range(I):
            for k in range(K):
                E_hi[i,k] = np.divide(nu_plus_D[k],nu_plus_delta[i,k])
               
        ## Maximization step.
        ## Update lambda
        for k in range(K):
            lam = np.sum(tau,axis=0) / I
           
        E_hi_times_tau = E_hi * tau
        E_hi_times_tau_sum = np.sum(E_hi_times_tau,axis = 0)
        new_mu = np.zeros([K,D])
        for k in range(K):
            for i in range(I):
                new_mu[k,:] = new_mu[k,:] + (E_hi_times_tau[i,k]*x[i,:])
            means[k,:] = np.divide(new_mu[k,:],E_hi_times_tau_sum[k])
    
        #Update sig.
        tau_sum_col = np.sum(tau,axis=0)
        for k in range(K):
            new_sigma = np.zeros([D,D])
            for i in range(I):
                mat = np.reshape((x[i,:] - means[k,:]),(1,-1))
                mat = E_hi_times_tau[i,k] * (np.transpose(mat) * mat)
                new_sigma = new_sigma + mat
            sig[k] = np.divide(new_sigma,tau_sum_col[k]);
            sig[k] = np.diag(np.diag(sig[k]))
                   
        ##Update nu by minimizing a cost function with line search.
        for k in range(K):      
            nu[k] = fit_tmix_cost_file.fit_tmix_cost(nu[k],E_hi[:,k],D)
    #   
        for i in range(I):
            for k in range(K):
#                inverse_sig = np.diag(1 / np.diag(sig[k]))
                delta[i,k] = (scipy.spatial.distance.mahalanobis(np.reshape(x[i,:],(1,-1)),means[k,:],np.linalg.inv(sig[k]))**2)
    #    temp = np.zeros([I,K]);
    #    l2 = np.zeros([I,K]);         
    #    for k in range(K):
    #        l2[:,k] = pdf_tm_file2.pdf_tm(x, means[k,:], sig[k],nu[k])
    #        temp[:,k] = lam[k] * l2[:,k]        
    #    temp = np.sum(temp,axis=1);
    #    temp = np.log(temp);        
    #    L = np.sum(temp,axis=0);  
    #    
        iterations = iterations + 1;
#        print(str(iterations)+' : '+str(L))
        print(str(iterations))
    #    if (np.absolute(L - previous_L) < precision) or 
        if iterations == 20:
            break
    #    previous_L = L;
    return(means,sig,nu,lam)
