import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
import image_array_create_2
import plot_mu_sig_3
import plot_roc_curve_4
import gom_em_5
import fit_t_file_6
import pdf_tm_file_8
import fit_fa_file_9
import mix_tfit_file_10
import pdf_tm_file2_11



rain_image_path = 'C:\\Users\\nazne\\OneDrive\\Documents\\ECE 763 Computer Vision\\Project\\Project1\\Project_try2\\images'
path_test = 'C:\\Users\\nazne\\OneDrive\\Documents\\ECE 763 Computer Vision\\Project\\Project1\\Project_try2\\Testing'
path_train = 'C:\\Users\\nazne\\OneDrive\\Documents\\ECE 763 Computer Vision\\Project\\Project1\\Project_try2\\Training_Images'

no_of_train = 1000
no_of_test = 100
image_size = 60
[t_test,u_test,t_train,u_train] = image_array_create_2.image_array_create(no_of_train,no_of_test,image_size,'GRAY')

print('Performing PCA for training and tetsing data...')       
pca_face = PCA(30)
t_train_pca = pca_face.fit_transform(t_train) 
u_train_pca = pca_face.fit_transform(u_train) 
t_test_pca  = pca_face.fit_transform(t_test)
u_test_pca  = pca_face.fit_transform(u_test) 
#print(X_proj.shape)
#print(np.cumsum(pca_face.explained_variance_ratio_))
   
########### Labels for ROC PLOT Generation ###########################################################
n1,m1 = t_test.shape 
n2,m2 = u_test.shape
t0 = np.ones((n1,1))
u0 = np.zeros((n2,1))
tnew = np.hstack((t_test,t0))
unew = np.hstack((u_test,u0))
X = np.concatenate((tnew, unew), axis=0)
np.random.shuffle(X[0:n1+n2])
labels = X[:,-1]
X_test_roc = X[:,:-1]
X_roc = pca_face.fit_transform(X_test_roc) 

##############################################################################
                           # Simple Gaussian Model
##############################################################################

## Calculating the means 
mu_face    = t_train_pca.mean(axis=0)
mu_nonface = u_train_pca.mean(axis=0)
# Calculating the co-variance and defining as a diagonal matrix
var_face = np.cov(t_train_pca, rowvar=False, bias=1, ddof=None)
var_face = np.diagonal(var_face)
var_face = np.diag(var_face,0)
var_nonface = np.cov(u_train_pca, rowvar=False, bias=1, ddof=None)
var_nonface = np.diagonal(var_nonface)
var_nonface = np.diag(var_nonface,0)

##mu_face_inv = pca_face.inverse_transform(mu_face.flatten())
##var_face_inv = pca_face.inverse_transform(np.diag(var_face))
##mu_nonface_inv = pca_face.inverse_transform(mu_nonface.flatten())
##var_nonface_inv = pca_face.inverse_transform(np.diag(var_nonface))

plot_mu_sig_3.plot_mu_sig(mu_face,np.diag(var_face),'Mean of Face Images','Variance of Face Images','NOT GRAY',image_size,'BOTH')
plot_mu_sig_3.plot_mu_sig(mu_nonface,np.diag(var_nonface),'Mean of Background Images','Variance of Background Images','NOT GRAY',image_size,'BOTH')

prob_fpf_Roc = multivariate_normal.pdf(X_roc, mean=mu_face, cov=var_face)
prob_fnpf_Roc = multivariate_normal.pdf(X_roc, mean=mu_nonface, cov=var_nonface)
total_Roc = prob_fpf_Roc + prob_fnpf_Roc
P_Roc = prob_fpf_Roc / total_Roc
plot_roc_curve_4.plot_roc_curve(labels,P_Roc,1,'Receiver Operating Characteristic for Simple Gaussian Model')

prob_fpf = multivariate_normal.pdf(t_test_pca, mean=mu_face, cov=var_face)
prob_fnpf = multivariate_normal.pdf(t_test_pca, mean=mu_nonface, cov=var_nonface)
total1 = prob_fpf + prob_fnpf
Pr1 = prob_fpf / total1
cor_face = np.sum(Pr1[:] >= 0.5)
noncor_face = 100 - cor_face 

prob_nfpf = multivariate_normal.pdf(u_test_pca, mean=mu_face, cov=var_face)
prob_nfnpf = multivariate_normal.pdf(u_test_pca, mean=mu_nonface, cov=var_nonface)
total2 = prob_nfpf + prob_nfnpf
Pr2 = prob_nfnpf / total2
cor_nonface = np.sum(Pr2[:] >= 0.5)
noncor_nonface = 100 - cor_nonface

FPR= noncor_nonface / (noncor_nonface + cor_nonface)
FNR =  noncor_face / (cor_face + noncor_face)
MCR = (noncor_nonface + noncor_face) / 200

print('False Positive Rate:' + str(FPR))
print('False Negative Rate:' + str(FNR))
print('Miss Classification Rate:' + str(MCR))

############################################################################
                             # Mixture of Gaussian
############################################################################
#Works for 9x9 Image size
print('Running Mixture of Gaussian Model...')
 
K = 10
(lam,means,sig) = gom_em_5.MOG_EM(t_train_pca,K)

X_proj_img = [[]]*K
for i in range(10):
    sig_new = np.diag(sig[i])
    X_inv_proj = pca_face.inverse_transform(sig_new) 
    X_proj_img[i] = np.reshape(X_inv_proj,(60,60))

#plot the faces, each image is 64 by 64 dimension but 8x8 pixels
#for i in range(5):
#     ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[]) 
#     ax.imshow(X_proj_img[i], cmap="Greys_r", interpolation='nearest')   
#
#k = 0
#for i in range(5,10):
#     ax = fig.add_subplot(8, 8, k+1, xticks=[], yticks=[]) 
#     ax.imshow(X_proj_img[i], cmap="Greys_r", interpolation='nearest')
#     k+=1

Prn = np.zeros([n1+n2,1])
sum1 = np.zeros([n1+n2,1])
for k in range(K):
    Prn = multivariate_normal.pdf(X_roc, means[k,:], sig[k],allow_singular=True)
    Prn = np.reshape(Prn,(-1,1))
    a = lam[k]
    sum1 = sum1 + (a * Prn)

sum2 = multivariate_normal.pdf(X_roc, mean=mu_nonface, cov=var_nonface,allow_singular=True)
Prn_whole_roc = sum1 / (sum1+ np.reshape(sum2,(n1+n2,1)))
plot_roc_curve_4.plot_roc_curve(labels,Prn_whole_roc,1,'Receiver Operating Characteristic for Mixture of Gaussian Model')


(lam2,means2,sig2) = gom_em_5.MOG_EM(t_train_pca,K)
Prn2 = np.zeros([100,1])
prob_fpf = np.zeros([100,1])
for k in range(K):
    Prn2 = multivariate_normal.pdf(t_test_pca, means2[k,:], sig2[k])
    Prn2 = np.reshape(Prn2,(-1,1))
    a = lam2[k]
    prob_fpf = prob_fpf + (a * Prn2)
prob_fpnf = multivariate_normal.pdf(t_test_pca, mean= mu_nonface, cov=var_nonface)
Prn_face = prob_fpf / (prob_fpf+ np.reshape(prob_fpnf,(100,1)))
cor_face_mog = np.sum(Prn_face[:] >= 0.5)
noncor_face_mog = 100 - cor_face_mog

(lam3,means3,sig3) = gom_em_5.MOG_EM(u_train_pca,K)
Prn3 = np.zeros([100,1])
prob_nfpnf = np.zeros([100,1])
for k in range(K):
    Prn3 = multivariate_normal.pdf(u_test_pca, means3[k,:], sig3[k])
    Prn3 = np.reshape(Prn3,(-1,1))
    a = lam3[k]
    prob_nfpnf = prob_nfpnf + (a * Prn3)
prob_nfpf = multivariate_normal.pdf(u_test_pca, mean= mu_face, cov=var_face)
Prn_nonface = prob_nfpnf / (prob_nfpnf+ np.reshape(prob_nfpf,(100,1)))
cor_nonface_mog = np.sum(Prn_nonface[:] >= 0.5)
noncor_nonface_mog = 100 - cor_nonface_mog

FPR= noncor_nonface_mog / (noncor_nonface_mog + cor_nonface_mog)
FNR =  noncor_face_mog / (cor_face_mog + noncor_face_mog)
MCR = (noncor_nonface_mog + noncor_face_mog) / 200

print('False Positive Rate:' + str(FPR))
print('False Negative Rate:' + str(FNR))
print('Miss Classification Rate:' + str(MCR))
###############################################################################
                                 # t DISTRIBUTION
###############################################################################
# works with 30 x 30 grey image
print('Call t distribution...')
(mu_f,sig_f,nu_f) = fit_t_file_6.fit_t(t_train_pca, 0.01)

#X_inv_proj = pca_face.inverse_transform(mu_f) 
#X_proj_img = np.reshape(X_inv_proj,(image_size,image_size))
#X_proj_img = X_proj_img / np.max(X_proj_img) 
#matplotlib.pyplot.imshow(X_proj_img,cmap="Greys_r")
#sig_f_new = np.diag(sig_f)
#X_inv_proj = pca_face.inverse_transform(sig_f_new) 
#X_proj_img = np.reshape(X_inv_proj,(image_size,image_size))
#X_proj_img = X_proj_img / np.max(X_proj_img) 
#plt.imshow(X_proj_img,cmap="Greys_r")

(mu_nf,sig_nf,nu_nf) = fit_t_file_6.fit_t(u_train_pca,0.01) 
pr1 = pdf_tm_file_8.pdf_tm(X_roc,mu_f,sig_f,nu_f)
pr2 = pdf_tm_file_8.pdf_tm(X_roc,mu_nf,sig_nf,nu_nf)
pr = pr1 /(pr1+pr2)
plot_roc_curve_4.plot_roc_curve(labels,pr,1,'Receiver Operating Characteristic for T-Distribution')
                            
px_face_pf = pdf_tm_file_8.pdf_tm(t_test_pca,mu_f,sig_f,nu_f)
px_nonface_pf = pdf_tm_file_8.pdf_tm(u_test_pca,mu_f,sig_f,nu_f)
px_face_pnf = pdf_tm_file_8.pdf_tm(t_test_pca,mu_nf,sig_nf,nu_nf)
px_nonface_pnf = pdf_tm_file_8.pdf_tm(u_test_pca,mu_nf,sig_nf,nu_nf)
Prob_face = px_face_pf / (px_face_pf + px_face_pnf)
Prob_nonface = px_nonface_pnf / (px_nonface_pnf + px_nonface_pf)

cor_face_t = np.sum(Prob_face[:] >= 0.5)
noncor_face_t = 100 - cor_face_t 
cor_nonface_t = np.sum(Prob_nonface[:] >= 0.5)
noncor_nonface_t = 100 - cor_nonface_t

TPR_t = cor_face_t / (cor_face_t + cor_nonface_t)
FPR_t = noncor_nonface_t / (noncor_nonface_t + cor_nonface_t)
TNR_t = cor_nonface_t / (noncor_nonface_t + noncor_face_t)
FNR_t =  noncor_face_t / (cor_face_t + noncor_face_t)
MR    = (noncor_nonface_t + noncor_face_t) / 200
print('False Positive Rate:' + str(FPR_t))
print('False Negative Rate:' + str(FNR_t))
print('Miss Classification Rate:' + str(MR))

###############################################################################
                        # Factor Analysis
###############################################################################
print('Perorming Factor Analysis...')
train = t_train
(I,D) = np.shape(train)
K = 4
iterations = 30
[mu, phi, sig] = fit_fa_file_9.fit_fa(train, K, iterations)   

train = u_train
(I,D) = np.shape(train)
[mu_nf, phi_nf, sig_nf] = fit_fa_file_9.fit_fa(train, K, iterations) 

var_face = np.dot(phi,np.transpose(phi)) + np.diag(sig)
var_face_nf = np.dot(phi_nf,np.transpose(phi_nf)) + np.diag(sig_nf)

row,col = X_test_roc.shape
prob_fpf_Roc = multivariate_normal.pdf(X_test_roc, mu.reshape((col,)), cov=var_face,allow_singular=True)
prob_fnpf_Roc = multivariate_normal.pdf(X_test_roc, mu_nf.reshape((col,)), cov=var_face_nf,allow_singular=True)
total_Roc = prob_fpf_Roc + prob_fnpf_Roc
P_Roc = prob_fpf_Roc / total_Roc
plot_roc_curve_4.plot_roc_curve(labels,P_Roc,1,'Receiver Operating Characteristic for Factor Analysis')

mu_pca = pca_face.inverse_transform(mu)
sig_mca = pca_face.inverse_transform(sig)
mu_pca = mu
sig_pca = sig
X_proj_img = np.reshape(mu_pca,(60,60))
X_proj_img = X_proj_img / np.max(mu_pca) 
plt.imshow(X_proj_img,cmap="Greys_r")

X_proj_img = np.reshape(sig_pca,(60,60))
X_proj_img = X_proj_img / np.max(sig_pca) 
plt.imshow(X_proj_img,cmap="Greys_r")
plot_mu_sig_3.plot_mu_sig (mu_pca,sig_pca,'Mean of Face Images','Variance of Face Images')

################################################################################################################

px_face_pf = multivariate_normal.pdf(t_test, mu.reshape((D,)), cov=var_face,allow_singular=True)
px_face_pnf = multivariate_normal.pdf(t_test, mu_nf.reshape((D,)), cov=var_face_nf,allow_singular=True)
total = px_face_pf + px_face_pnf
Prob_face = px_face_pf / total

px_nonface_pf = multivariate_normal.pdf(u_test, mu.reshape((D,)), cov=var_face,allow_singular=True)
px_nonface_pnf = multivariate_normal.pdf(u_test, mu_nf.reshape((D,)), cov=var_face_nf,allow_singular=True)
total = px_nonface_pf + px_nonface_pnf
Prob_nonface = px_nonface_pnf/ total

cor_face_t = np.sum(Prob_face[:] >= 0.5)
noncor_face_t = no_of_test - cor_face_t 
cor_nonface_t = np.sum(Prob_nonface[:] >= 0.5)
noncor_nonface_t = no_of_test - cor_nonface_t
#
TPR_t = cor_face_t / (cor_face_t + cor_nonface_t)
FPR_t = noncor_nonface_t / (noncor_nonface_t + noncor_face_t)
TNR_t = cor_nonface_t / (noncor_nonface_t + noncor_face_t)
FNR_t =  noncor_face_t / (cor_face_t + cor_nonface_t)
MR = (noncor_nonface_t + noncor_face_t) / no_of_test

print('False Positive Rate:' + str(FPR_t))
print('False Negative Rate:' + str(FNR_t))
print('Miss Classification Rate:' + str(MR))

#Example similar to (Figure 7.22) in Dr. Prince's book.
rate = 0.0001
mu_phi_lin_combinations = np.zeros([8,D]);

new1 = np.zeros([1,len(mu)])
new2 = np.zeros([1,len(mu)])
for i in range(0,4):
    phi_ = np.transpose(phi[:,i])
#    % Go away from mu in positive phi_ direction.
    v = mu
    new1 = v + rate * phi_;      
    mu_phi_lin_combinations[i,:] = new1
#    % Go away from mu in negative phi_ direction.
    new2 = v - rate * phi_     
    mu_phi_lin_combinations[4+i,:] = new2
#
#% Reshape and normalize.
mu_phi_lin_combinations_mat = [[]]*8

    #X_inv_proj = pca_face.inverse_transform(sig_f_new) 

for i in range(8):
    mp = pca_face.inverse_transform(mu_phi_lin_combinations[i,:])
    mp = mp / np.max(mp)
    mp = np.reshape(mp,(image_size,image_size))
    mu_phi_lin_combinations_mat[i] = mp
#% Plot the linear combinations between mu and the phi column vectors.

for i in range(8):
    plt.imshow(mu_phi_lin_combinations_mat[i],cmap='gray')
    plt.show()
###############################################################################
                        # Mixture of t Distribution
###############################################################################
x = t_train_pca
K = 3
[means_f,sig_f,nu_f,lam_f] = mix_tfit_file_10.fit_mix_t(x, 0.01,K)
x = u_train_pca
[means_nf,sig_nf,nu_nf,lam_nf] = mix_tfit_file_10.fit_mix_t(x, 0.01,K)
for k in range(K):
   m = pca_face.inverse_transform(means_f[k]) 
   s = pca_face.inverse_transform(np.diag(sig_f[k])) 
   
   X_proj_img = np.reshape(m,(60,60))
   X_proj_img = X_proj_img / np.max(m) 
   X_proj_img1 = np.reshape(s,(60,60))
   X_proj_img1 = X_proj_img1 / np.max(s)
   plt.imshow(X_proj_img,cmap='gray')   
   plt.show()
   plt.imshow(X_proj_img1,cmap='gray')    
   plt.show()
   
   plot_mu_sig_3.plot_mu_sig(m,(s),'Mean of Face Images_'+str(k),'Variance of Face Images_'+str(k),'GRAY',image_size,'BOTH')

######################################## ROC PLOT #############################

I,D = X_roc.shape
temp = np.zeros([I,K])
l2 = np.zeros([I,K])  

for k in range(K):
    l2[:,k] = pdf_tm_file2_11.pdf_tm(X_roc, means_f[k,:], sig_f[k],nu_f[k])
    a = lam_f[k]
    temp[:,k] = a * l2[:,k];   
Pr_face = np.sum(temp,axis=1);

temp = np.zeros([I,K]);
l2 = np.zeros([I,K]);  
for k in range(K):
    l2[:,k] = pdf_tm_file2_11.pdf_tm(X_roc, means_nf[k,:], sig_nf[k],nu_nf[k])
    a = lam_nf[k]
    temp[:,k] = a * l2[:,k];           
Pr_noface = np.sum(temp,axis=1);

#Pr_noface = multivariate_normal.pdf(X[:,:-1], mean=mu_nonface, cov=var_nonface)
P_Roc = Pr_face / (Pr_face + Pr_noface)
plot_roc_curve_4.plot_roc_curve(labels,P_Roc,1,'Receiver Operating Characteristic for Mixture of T model')
###############################################################################

I,D = t_test_pca.shape
temp = np.zeros([I,K]);
l2 = np.zeros([I,K]);  
for k in range(K):
    l2[:,k] = pdf_tm_file2_11.pdf_tm(t_test_pca, means_f[k,:], sig_f[k],nu_f[k])
    a = lam_f[k]
    temp[:,k] = a * l2[:,k];   
px_face_pf = np.sum(temp,axis=1);

temp = np.zeros([I,K]);
l2 = np.zeros([I,K]);  
for k in range(K):
    l2[:,k] = pdf_tm_file2_11.pdf_tm(t_test_pca, means_nf[k,:], sig_nf[k],nu_nf[k])
    a = lam_nf[k]
    temp[:,k] = a * l2[:,k];           
px_face_pnf = np.sum(temp,axis=1)
total = px_face_pf + px_face_pnf
Prob_face = px_face_pf / total

I,D = u_test_pca.shape
temp = np.zeros([I,K]);
l2 = np.zeros([I,K]);  
for k in range(K):
    l2[:,k] = pdf_tm_file2_11.pdf_tm(t_test_pca, means_f[k,:], sig_f[k],nu_f[k])
    a = lam_f[k]
    temp[:,k] = a * l2[:,k];   
px_nonface_pf = np.sum(temp,axis=1);
 
temp = np.zeros([I,K]);
l2 = np.zeros([I,K]);  
for k in range(K):
    l2[:,k] = pdf_tm_file2_11.pdf_tm(t_test_pca, means_nf[k,:], sig_nf[k],nu_nf[k])
    a = lam_nf[k]
    temp[:,k] = a * l2[:,k];           
px_face_pnf = np.sum(temp,axis=1)
total = px_face_pf + px_face_pnf
px_nonface_pnf = px_face_pf / total
total = px_nonface_pf + px_nonface_pnf
Prob_nonface = px_nonface_pnf/ total

cor_face_t = np.sum(Prob_face[:] >= 0.5)
noncor_face_t = no_of_test - cor_face_t 
cor_nonface_t = np.sum(Prob_nonface[:] >= 0.5)
noncor_nonface_t = no_of_test - cor_nonface_t

FPR_t = noncor_nonface_t / (noncor_nonface_t + cor_nonface_t)
FNR_t =  noncor_face_t / (cor_face_t + noncor_face_t)
MR = (noncor_nonface_t + noncor_face_t) / no_of_test

print('False Positive Rate:' + str(FPR_t))
print('False Negative Rate:' + str(FNR_t))
print('Miss Classification Rate:' + str(MR))
                      
