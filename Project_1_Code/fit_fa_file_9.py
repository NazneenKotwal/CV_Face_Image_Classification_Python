# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 09:45:10 2018
@author: Nazneen Kotwal
"""

#% Author: Stefan Stavrev 2013
#
#% Description: Fitting a factor analyzer.
#% Input: x - a matrix where each row is one datapoint,
#%        K - number of factors.
#% Output: mu - 1xD mean vector,
#%         phi - DxK matrix containing K factors in its columns,
#%         sig - Dx1 vector representing the DxD diagonal matrix Sigma.

import numpy as np 
def fit_fa (X, K, iterations):
#    K = 6
#    N = 10;
#    data_mu = [1 ,2]
#    data_sig = [[2, 0], [0, .5]]
#    X = np.random.multivariate_normal(data_mu, data_sig,N);
    
    (I,D) = np.shape(X)
    # Initialize mu to the data mean.    
    dataset_mean = X.mean(axis=0)
    mu = np.reshape(dataset_mean,(1,-1))
    
    #    %Initialize phi to random values.
    #    rng('default');
    #    phi = randn(D,K);
    np.random.seed(0)
    phi = np.random.randn(D,K)
    
    #    % Initialize sig, by setting its diagonal elements to the
    #    % variances of the D data dimensions.
    #    x_minus_mu = bsxfun (@minus, X, mu);
    
    x_minus_mu = np.subtract(X, dataset_mean)
    sig = np.sum((x_minus_mu)**2, axis = 0) / I;
    
    #% The main loop.
    iterations_count = 0
    while True:
    #    % Expectation step.
        inv_sig = np.diag(1 / sig)
        phi_transpose_times_sig_inv = np.dot(np.transpose(phi),inv_sig)
        temp = np.linalg.inv(np.dot(phi_transpose_times_sig_inv,phi) + np.identity(K))
        E_hi = np.dot(np.dot(temp,phi_transpose_times_sig_inv),np.transpose(x_minus_mu))
        E_hi_hitr = [[]]*I
        for i in range (I):
            e = E_hi[:,i]
            E_hi_hitr[i] = temp + np.dot(e,np.transpose(e))
        
    #    % Maximization step.
    #    % Update phi.
        phi_1 = np.zeros([D,K])
        for i in range(I):
            sub1 = np.transpose(np.reshape(x_minus_mu[i,:],(1,-1)))
    #        sub2 = np.transpose(np.reshape(E_hi[:,i],(-1,1)))
            sub2 = np.transpose(np.reshape(E_hi[:,i],(-1,1)))
            phi_1 = phi_1 + np.dot(sub1,sub2)
        
        phi_2 = np.zeros([K,K])
        for i in range(I):
            phi_2 = phi_2 + E_hi_hitr[i]
        phi_2 = np.linalg.inv(phi_2)
        phi = np.dot(phi_1,phi_2)
    
    #    % Update sig.        
        sig_diag = np.zeros([D,1])
        for i in range(I):
            xm = np.transpose(x_minus_mu[i,:])
            sig_1 = xm * xm
            sig_2 = np.dot(phi,E_hi[:,i]) * xm;
            sig_diag = sig_diag + sig_1 - sig_2
        sig = sig_diag / I
        sig = np.diag(sig)
        
        iterations_count = iterations_count + 1    
        print(str(iterations_count))          
        if iterations_count == iterations:
            break;
    return(mu, phi, sig)