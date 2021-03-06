# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 16:12:48 2018

@author: Nazneen Kotwal

"""
import numpy as np
from scipy.special import gammaln
nu = np.arange(1,1000)
val_test = np.zeros([len(nu)])
# The cost function to be minimized for nu.
#def fit_t_cost (nu, E_hi, E_log_hi):
def fit_t_cost (E_hi, E_log_hi):
 for i in range(len(nu)):
    nu_half = nu[i] / 2
    (I,m) = np.shape(E_hi)
    val = I * (nu_half * np.log(nu_half) + gammaln(nu_half))
    val = val - (nu_half-1)* np.sum(E_log_hi)
    val = val + nu_half* np.sum(E_hi)
    val_test[i] = -1 * val # the minus in front of the main sum
 a = np.where(val_test == val_test.min())
 return (nu[a])
