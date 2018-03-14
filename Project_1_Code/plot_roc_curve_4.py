# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 00:00:25 2018

@author: Nazneen Kotwal
"""
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn import metrics

def plot_roc_curve(labels,P_Roc,pos,title):
    fpr, tpr, thresholds = metrics.roc_curve(labels,P_Roc, pos_label= pos)
    roc_auc = roc_auc_score(labels,P_Roc)
    # Plot ROC curve
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate or (1 - Specifity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    plt.title(title)
    plt.legend(loc="lower right")