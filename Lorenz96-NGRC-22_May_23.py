# -*- coding: utf-8 -*-
"""
Created on Jan 2022

The Ohio State University
Department of Physics
@author: Wendson Antonio de Sa Barbosa

This code reproduces the results presented in our paper:
    Learning Spatiotemporal Chaos using Next-Generation Reservoir Computing
    Wendson A. S. Barbosa and Daniel J. Gauthier
    Chaos 32, 093137 (2022) https://doi.org/10.1063/5.0098707
    Preprint also available at: https://arxiv.org/abs/2203.13294

Last modified: May, 22, 2023: 
    -Organized some functions and commments for a initial shareable version.
    -Added option to integrate L96 system rather than load data
"""

import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.preprocessing import PolynomialFeatures
import copy
import sklearn.linear_model
import sys


# =============================================================================
# Import data for the learning system - See folder "" for data files
# =============================================================================

# These data files where obtained integrating Eq. (1) of the paper 
# using fourth order Runge-Kutta method. 

# Select the learning system by difficulty:
#   "hard":     LorenzL6 with L=36, I=J=10
#   "medium":   LorenzL6 with L=I=J=8
#   "easy":     LorenzL6 with L=40, I=J=0

# For more details on the these Lorenz96 variants, please refer to our paper:
# Chaos 32, 093137 (2022) https://doi.org/10.1063/5.0098707
    
L96system = "hard"  # or "medium" or "easy"  

dt = 0.001
t_total = 100

integrate = True

if integrate == False:
    t_total = 11000 # size of data to be loaded 

if L96system == "hard":

    # Number of variables
    K = 36; J = 10; L = 10
    # System parameters
    F = 20; h = 1; c = 10; b = 10; e = 10; d = 10; g = 10
    # Lyapunov time for normalization of the time variable
    lyapunov_time = None # We do not normalize the time for this case  

elif L96system == "medium":

    # Number of variables
    K = 8; J = 8; L = 8
    # System parameters
    F = 20; h = 1; c = 10; b = 10; e = 10; d = 10; g = 10
    # Lyapunov time for normalization of the time variable
    lyapunov_time = None # We do not normalize the time for this case  
    
elif L96system == "easy":

    # Number of variables
    K = 40; J = 0; L = 0
    # System parameters
    F = 8; h = 0; c = 0; b = 0; e = 0; d = 0; g = 0
    # Lyapunov time for normalization of the time variable
    lyapunov_time = 1/1.68 

else:
    
    print("choose one of the L96 cases: 'hard', 'medium' or 'easy'")
    sys.exit()

# Just to make the code consistent
if lyapunov_time == None:
    lyapunov_time = 1

# IMPORTANT: We just use the variables X, which have the coarsest spatiotemporal scale,
#            although we have integrated the full set of equations (Eq. 1 of the paper.)

if integrate:
    
    # array with integration time 
    t_original = np.arange(0, t_total, dt)

    # =============================================================================
    # Runge-Kutta 4th order method
    # =============================================================================

    # L96 equations

    def dX_dt(X, Y, Z):
        dXdt = (np.roll(X, -1) * (np.roll(X, 1) -np.roll(X, -2)) - X + F - (h * c / b) * Y.reshape(K, J).sum(1))
        return dt * dXdt

    def dY_dt(X, Y, Z):
        dYdt = (- c * b * np.roll(Y, +1) * (np.roll(Y, +2) - np.roll(Y, -1)) - c * Y + (h * c / b) * np.repeat(X, J) - (h * e / d) * Z.reshape(K,J,L).sum(2).flatten())
        return dt * dYdt

    def dZ_dt(X, Y, Z):
        dZdt = ( e * d * np.roll(Z, -1) * (np.roll(Z, +1) - np.roll(Z, -2)) - g * e * Z + (h * e / d) * np.repeat(Y, L))
        return dt * dZdt


    # RK4 step function

    def RK_step(X,Y,Z):


        k1_X = dX_dt(X, Y, Z)
        k1_Y = dY_dt(X, Y, Z)
        k1_Z = dZ_dt(X, Y, Z)
        
        k2_X = dX_dt(X + k1_X / 2, Y + k1_Y / 2, Z + k1_Z / 2)
        k2_Y = dY_dt(X + k1_X / 2, Y + k1_Y / 2, Z + k1_Z / 2)
        k2_Z = dZ_dt(X + k1_X / 2, Y + k1_Y / 2, Z + k1_Z / 2)
        
        k3_X = dX_dt(X + k2_X / 2, Y + k2_Y / 2, Z + k2_Z / 2)
        k3_Y = dY_dt(X + k2_X / 2, Y + k2_Y / 2, Z + k2_Z / 2)
        k3_Z = dZ_dt(X + k2_X / 2, Y + k2_Y / 2, Z + k2_Z / 2)

        k4_X = dX_dt(X + k3_X, Y + k3_Y, Z + k3_Z)    
        k4_Y = dY_dt(X + k3_X, Y + k3_Y, Z + k3_Z)
        k4_Z = dZ_dt(X + k3_X, Y + k3_Y, Z + k3_Z)

        X += 1 / 6 * (k1_X + 2 * k2_X + 2 * k3_X + k4_X)
        Y += 1 / 6 * (k1_Y + 2 * k2_Y + 2 * k3_Y + k4_Y)
        Z += 1 / 6 * (k1_Z + 2 * k2_Z + 2 * k3_Z + k4_Z)
        
        return (X,Y,Z)
        

    # =============================================================================
    # Integration    
    # =============================================================================

    # Initial Conditions    

    X_ant = F*np.ones((K))
    X_ant[0] += 0.01 
    Y_ant = np.zeros((K*J))
    Z_ant = np.zeros((K*J*L))

    start = time.time()
    x_original = []
    y_original = []
    z_original = []

    for i in range(len(t_original)):
        X,Y,Z=RK_step(X_ant,Y_ant,Z_ant)
        X_ant = copy.deepcopy(X)
        Y_ant = copy.deepcopy(Y)
        Z_ant = copy.deepcopy(Z)
        
        # If want to decimate saved data, need to define dt_eval
        #if i%(int (dt_eval/dt))==0:
        
        # Usually I only keep X variables to save memory.
        x_original.append(X)
        #y_original.append(Y)
        #z_original.append(Z)


    print("Time for Lorenz96 integration with RK: ", time.time() -start)

    x_original = np.asarray(x_original).T    
    
else:
    x_original,t_original = np.load("Data_and_code_for_sharing/L96-Ttotal_%d-F_%d-K_%d-J_%d-L_%d-h_%d-b_%d-c_%d-d_%d-e_%d-g_%d.npy" %(t_total,F,K,J,L,h,b,c,d,e,g),allow_pickle=True)
 
   
##Discarding 10000 transient points (dt=0.001, 10 MTU)
x_original = x_original[:,10000:]
t_original = t_original[10000:]


# =============================================================================
# Renaming, Rescaling and Normalizing data
# =============================================================================

# Just a change in the name of the variable to be more appropriate for the code
N=K

##Rescaling/Normalizing 
x_mean = np.mean(x_original)
x_std = np.std(x_original)

x = x_original - x_mean
x = x/x_std

dt_eval = 0.01 #sampling time for data; and also prediction every dt_eval

t = t_original[::int(dt_eval/dt)]
x = x[:,::int(dt_eval/dt)]

# Delete original variables to save memory
del x_original, t_original


'''
Change the lines after this comment to reproduce figures of the paper.
Examples:
    
    1.If L96system is chosen "hard" in the lines above, then Fig. 2 is obtained using:
        t_transient = 10 
        t_training = 10 
        t_testing = 5 
        Num_Pred = 1 
        Parallel = False
        Respect_Symmetry = False
        poly_degree = 2
        k=3
        s=1 
        ridge_param = 1e-2
        
    2.If L96system is chosen "hard" in the lines above, then Fig. 3 is obtained using:
        t_transient = 10 
        t_training = 10 
        t_testing = 5 
        Num_Pred = 1 
        Parallel = True
        Respect_Symmetry = False
        poly_degree = 2
        k=3
        s=1 
        ridge_param = 1e-2
    
    3.If L96system is chosen "hard" in the lines above, then Fig. 4 is obtained using:
        t_transient = 10 
        t_training = 10 
        t_testing = 5 
        Num_Pred = 1 
        Parallel = True
        Respect_Symmetry = True
        poly_degree = 2
        k=3
        s=1 
        ridge_param = 1e-2

    For the figures like Fig. 7 and Fig. 9, change L96system to "medium" or "easy", respectively.

    For Figs. 5,6 and 8, run this code for different transients (which means different training data sets)
    and for Num_Pred > 1 (different predictions) and plot as function of training sizes (t_training). 
    
    For Fig. 10, perform a grid search on ridge_param for different values of t_training.
'''

# =============================================================================
# Training and Prediction parameters     
# =============================================================================

t_transient = 10 #Transient time
t_training = 10 #Training time    
t_testing = 5 #Prediction time
Num_Pred = 1 #Number of predictions

# Regularization parameter
ridge_param = 1e-2

# =============================================================================
# NG-RC architecture
# =============================================================================

# Using parallel NG-RCs? If false, use a single NG-RC to predict all variables
Parallel = True
# If Respect_Symmetry is True, train a single NG-RC using all data
Respect_Symmetry = True
# Polynomial degree for the nonlinear features
poly_degree = 2
poly = PolynomialFeatures(poly_degree) 
k=3 # current variables and k-1 tap delays
s=1 # skip parameter - for this paper, we kept it = 1 (don't skip delay steps)




# DO NOT NEED TO CHANGE PARAMETERS BELOW THIS LINE TO REPRODUCE THE PAPER FIGURES




# Default for Parallel = False
Nrc = 1 # Number of NG-RCs
NN = None # Num. of Nearest Neighbohrs nodes (each side) from where NG-RC gets data 
Npred = N # Number of nodes predicted by each NG-RC
Lrc = N # Length of each NG-RC unit (Number total of nodes where it gets data from)
N_overlay = 0 # Just a variable that helps to create the feature vector properly

# If parallel, set number of NG-RCs "Nrc" as the dimension of the learning system
if Parallel:
    Nrc = N 
    NN = 2
    Lrc = 1 + NN*2      
    Npred = N/Nrc            
    N_overlay = 2*NN
    

#Number of spatial locations (variables) to train a single NG-RC respecting symmetry
NumLocations = N 
#Get first NumLocations variables 
locations = list(np.arange(0,NumLocations,1)) 
 

# =============================================================================
# Features creation and training
# =============================================================================

global_time_start = time.time()


# Indexes of previous steps to be used in training/prediction. 
list_index = []
for j in range(k):
    list_index.append(-j*s-1)



# This function transform the input data of shape (N,time_steps) into a list 
# of Nrc Arrays each of size (Lrc,time_steps) and also return ground truth
def transform_training (x,Nrc,Lrc):

    x_rolled = np.roll(x,N_overlay//2,axis=0)
    if N_overlay > 0:    
        xextended = np.append(x_rolled,x_rolled[0:N_overlay,:],axis=0)
    else:
        xextended = x_rolled

    x_i = []
    True_training_i = []
    for Nrc_i in range(Nrc):
        #x_ = np.roll(xextended,-int(Nrc_i*(Lrc - N_overlay)),axis=0)[0:Lrc,:]
        x_ = xextended[int(Nrc_i*(Lrc - N_overlay)):int(Nrc_i*(Lrc - N_overlay))+Lrc,:]
        x_i.append(x_)
        
        if NN != None:
            True_training_i.append(x_[N_overlay//2:-N_overlay//2,int(t_transient/dt_eval)+1:int(t_transient/dt_eval)+1 + int(t_training/dt_eval)-1])
        else:
            True_training_i.append(x_[0:int(Npred) , int(t_transient/dt_eval)+1:int(t_transient/dt_eval)+1 + int(t_training/dt_eval)-1])    
    return (x_i,True_training_i)

x_i,True_training_i = transform_training (x,Nrc,Lrc)

#%

time_start_train = time.time()
# Defining some variables
output_model_i = Nrc*[0]
Regression_time = []
total_time_to_create_features = []

# If respecting symmetry, concatenate true data from all "locations" for training a single NG-RC
if Parallel and Respect_Symmetry:    
    features_all_train_all = []
    True_training_Nrc_all = np.concatenate([True_training_i[index] for index in locations]).flatten()

# =============================================================================
# Training
# =============================================================================
for Nrc_i in range(Nrc):
    
    x_Nrc = x_i[Nrc_i]
    True_training_Nrc = True_training_i[Nrc_i]  
    
    start_training_time = time.time()
    
    # Transforming data to include past values according to "k" and "s".
    def data_creation_training(t_training,k,s,x_Nrc,t_transient):
        data =[]
        for l in range(int(t_training/dt_eval)-1):
            data_aux = x_Nrc[:,int(t_transient/dt_eval)+l-k*s+1:int(t_transient/dt_eval)+l+1][:,list_index].flatten()
            data.append(data_aux)           
        return data    
    data_train = data_creation_training(t_training,k,s,x_Nrc,t_transient)


    # Creating Polynomial Features
    def create_features_vector(poly_degree,data):
          
        poly = PolynomialFeatures(poly_degree)
    
        features_all = poly.fit_transform(data)     
    
        return features_all
    
    start_feature_time = time.time()
    features_all_train = create_features_vector(poly_degree,data_train)
    
    # If respecting symmetry, append features from all "locations" for training a single NG-RC
    if Parallel and Respect_Symmetry:   
        if Nrc_i in locations:
            features_all_train_all.append(features_all_train)
    
    time_features_i = time.time() - start_feature_time
    total_time_to_create_features.append(time_features_i)   
    print('time for creating features for NG-RC # %d: ' % Nrc_i, time_features_i)
    

    # Training   
    start_training_time = time.time()
    
    output_model_i[Nrc_i] = sklearn.linear_model.Ridge(alpha=ridge_param)
    output_model_i[Nrc_i].fit(features_all_train,True_training_Nrc.T)
    
    training_time_i = time.time() - start_training_time
    Regression_time.append(training_time_i)
    print('time for training NG-RC # %d: ' % Nrc_i, time.time() - start_training_time)
    
#
start_training_time = time.time()
# Training for the case where a single NG-RC is trained respecting symmetry
if Parallel and Respect_Symmetry: 
    features_all_train_all = np.concatenate(features_all_train_all)
    output_model = sklearn.linear_model.Ridge(alpha=ridge_param)
    output_model.fit(features_all_train_all,True_training_Nrc_all.T)
print('time for training 1 NG-RC: ' , time.time() - start_training_time)    

total_time = time.time() - time_start_train

time_to_train = np.sum(Regression_time)

print('')
print('Time to create features for %d NG-RCs: '  % Nrc, np.sum(total_time_to_create_features))
print('Time to train %d NG-RCs: ' % Nrc, time_to_train)
print('Total time: ', total_time)
    

## Coefficients and Intercepts for the NG-RCs
Wouts = []
intercepts = []
for i in range(Nrc):
    #Wouts.append(output_model_i[i].coef_[0]) 
    if Parallel:
        Wouts.append(output_model_i[i].coef_[0]) 
    else:
        Wouts.append(output_model_i[i].coef_) #WIP
    intercepts.append(output_model_i[i].intercept_)

Wouts = np.asarray(Wouts)
intercepts = np.asarray(intercepts)

# Making Nrc copies of the single NG-RC when symmetry is respected
if Parallel and Respect_Symmetry:
    for i in range(Nrc):
        Wouts[i]= output_model.coef_    
        intercepts[i]= output_model.intercept_


# =============================================================================
# Metric to quantify how different Wouts are
# =============================================================================

if len(Wouts)>1:

    dot_all = []
    for i in range(len(Wouts)):
        dot_i = []
        for j in range(len(Wouts)):
            if j!=i:
                dot_i.append(np.dot(Wouts[i],Wouts[j]))
            if j==i:
                dot_self = np.dot(Wouts[i],Wouts[j])
        dot_all.append(np.mean(dot_i)/dot_self)
    
    
    metric = np.mean(dot_all)
#%

# =============================================================================
# Prediction
# =============================================================================

# Transform input data like the function "transform_testing", but for prediction phase.
def transform_testing(x_last):
    global N

    x_last_2 = copy.deepcopy(x_last[:,list_index])
    x_rolled = np.roll(x_last_2,N_overlay//2,axis=0)
    xextended = np.append(x_rolled,x_rolled[0:N_overlay,:],axis=0)

    x_i = []
    for Nrc_i in range(Nrc):
        x_i.append(xextended[int(Nrc_i*(Lrc - N_overlay)):int(Nrc_i*(Lrc - N_overlay)) + Lrc,:].flatten())
    return (x_i)


x_rolled = np.roll(x,N_overlay//2,axis=0)
if N_overlay > 0:    
    xextended = np.append(x_rolled,x_rolled[0:N_overlay,:],axis=0)
else:
    xextended = x_rolled

del x_rolled

# Initial times where predictions start (plural if Num_Pred is > 1).
t_i_range = np.linspace(0,(Num_Pred-1)*t_testing,Num_Pred)

NRMSE_i = []
e_i = []
error = []
prediction_time_all = []

# Loop for different "Num_Pred" predictions with different initial conditions
for t_i in t_i_range:

    t_i_index = int(t_transient/dt_eval)+1 + int(t_training/dt_eval)-1 + int(t_i/dt_eval -1)
    t_f_index = int(t_transient/dt_eval)+1 + int(t_training/dt_eval)-1 + int(t_i/dt_eval -1) + int(t_testing/dt_eval)
    
    # Loop to obtain ground truth for each of the N nodes
    True_testing_i = []
    x_i = []
    for Nrc_i in range(Nrc):
        x_ = np.roll(xextended,-int(Nrc_i*(Lrc - N_overlay)),axis=0)[0:Lrc,:]
        x_i.append(x_)
        
        if NN != None:
            True_testing_i.append(x_[N_overlay//2:-N_overlay//2,t_i_index:t_f_index])
        else:
            True_testing_i.append(x_[0:int(Npred) ,t_i_index:t_f_index])

    # Initial condition for prediction
    x_last = x[:,t_i_index-(s*(k-1)+1):t_i_index]
    start_prediction = time.time()
    testing_prediction = []
    
    for i in range(np.shape(True_testing_i)[-1]):
            
        x_last_i = transform_testing(x_last)  
        features_all_test = create_features_vector(poly_degree,x_last_i) 
        
        if Parallel:
            testing_prediction_i2 = np.asarray([np.dot(features_all_test[j], Wouts[j].T) + intercepts[j] for j in range(Nrc)])    
        else:
            testing_prediction_i2 = np.dot(features_all_test, Wouts[0].T) + intercepts    
        testing_prediction.append(testing_prediction_i2.reshape(-1))   
    
        x_last = np.delete(x_last,0,axis=1)
        x_last = np.hstack((x_last,testing_prediction_i2.reshape(-1,1))) 
    
    time_to_predict = time.time() - start_prediction
    print('')
    print("Time for prediction: ", time_to_predict)
    prediction_time_all.append(time_to_predict)

    
    True_testing = np.asarray(True_testing_i).squeeze()#[:,0,:]
    testing_prediction=np.asarray(testing_prediction)
    
    # Calculating NRMSE
    True_SpatialSTD = np.std(True_testing, axis=1) 
    SE = (True_testing - testing_prediction.T)**2 
    NSE = np.zeros((np.shape(SE)))
    for i in range(N):
        NSE[i] = SE[i]/((True_SpatialSTD [i])**2)
    
    NRMSE=np.sqrt(np.sum(NSE,axis=0)/(N))
    NRMSE_i.append(NRMSE)

#%
# =============================================================================
# Calculating Prediction Horizon
# =============================================================================

# Defining Plotting time
plot_time = t[int((t_transient+t_training)/dt_eval):int((t_transient+t_training+t_testing)/dt_eval)]/lyapunov_time - t[int((t_transient+t_training)/dt_eval)] /lyapunov_time

prediction_horizon_i = []
for i in range(len(NRMSE_i)):
    prediction_horizon_i.append(plot_time[np.where(NRMSE_i[i]> 0.3)[0][0]])

predict_horizon_mean = np.mean(prediction_horizon_i) 
predict_horizon_std = np.std(prediction_horizon_i) 
print('')
print('')
print("predict_horizon_mean: ", predict_horizon_mean)   
print("predict_horizon_std: ", predict_horizon_std) 

# Finding which prediction among the "Num_Pred"  returned the best prediction horizon 
predict_horizon_window_index= np.argmax(prediction_horizon_i) 
t_i_best = t_i_range[predict_horizon_window_index]


# =============================================================================
# Remake (for plotting) calculation for the best case among the "Num_Pred"  predictions
# =============================================================================

t_i=t_i_best
    
t_i_index = int(t_transient/dt_eval)+1 + int(t_training/dt_eval)-1 + int(t_i/dt_eval -1)
t_f_index = int(t_transient/dt_eval)+1 + int(t_training/dt_eval)-1 + int(t_i/dt_eval -1) + int(t_testing/dt_eval)


#True testing   
True_testing_i = []
x_i = []
for Nrc_i in range(Nrc):
    x_ = np.roll(xextended,-int(Nrc_i*(Lrc - N_overlay)),axis=0)[0:Lrc,:]
    x_i.append(x_)
    
    if NN != None:
        True_testing_i.append(x_[N_overlay//2:-N_overlay//2,t_i_index:t_f_index])
    else:
        True_testing_i.append(x_[0:int(Npred) ,t_i_index:t_f_index])

x_last = x[:,t_i_index-(s*(k-1)+1):t_i_index]

start_prediction = time.time()
testing_prediction = []
for i in range(np.shape(True_testing_i)[-1]):
        
    x_last_i = transform_testing(x_last)  #0.07
    features_all_test = create_features_vector(poly_degree,x_last_i)  #0.15
    
    #testing_prediction_i2 = np.asarray([np.dot(features_all_test[j], Wouts[j]) + intercepts[j] for j in range(Nrc)])    
    #testing_prediction.append(testing_prediction_i2.reshape(-1))   

    if Parallel:
        testing_prediction_i2 = np.asarray([np.dot(features_all_test[j], Wouts[j].T) + intercepts[j] for j in range(Nrc)])    
    else:
        testing_prediction_i2 = np.dot(features_all_test, Wouts[0].T) + intercepts    
    testing_prediction.append(testing_prediction_i2.reshape(-1))   


    x_last = np.delete(x_last,0,axis=1)
    x_last = np.hstack((x_last,testing_prediction_i2.reshape(-1,1))) 

time_to_predict = time.time() - start_prediction

True_testing = np.asarray(True_testing_i).squeeze()#[:,0,:]
testing_prediction=np.asarray(testing_prediction)


True_SpatialSTD = np.std(True_testing, axis=1) # np.max(True_testing, axis=0) - np.min(True_testing, axis=0)#
SE = (True_testing - testing_prediction.T)**2 
NSE = np.zeros((np.shape(SE)))
for i in range(N):
    NSE[i] = SE[i]/((True_SpatialSTD [i])**2)

NRMSE=np.sqrt(np.sum(NSE,axis=0)/N)


# =============================================================================
# Plot
# =============================================================================

fontsize_ = 20
labelsize_ = 20
title_size_ = 15
labelsize_clb = 20

fig, ax = plt.subplots(4,1,figsize=(10,7),sharex=True,gridspec_kw={'left':0.10,'right':0.91,'top':0.985,'bottom':0.1,'hspace': 0.2, 'wspace': 0})
fig.tight_layout()

im1 = ax[0].imshow(True_testing,aspect = "auto", cmap="jet", 
                    extent=[0,t_testing/lyapunov_time,0,N]
                    )

ax[0].set_ylabel("$x$", fontsize = fontsize_)
ax[0].tick_params(axis='both', which='major', labelsize=labelsize_)
ax[0].set_yticks(ticks=[0,N])
ax[0].set_yticklabels(["$x_1$","$x_L$"])

im2= ax[1].imshow(testing_prediction.T,aspect = "auto", cmap="jet", 
                  extent=[0,t_testing/lyapunov_time,0,N]
                  )
ax[1].set_ylabel("$\overline {x}$", fontsize = fontsize_)
ax[1].tick_params(axis='both', which='major', labelsize=labelsize_)
ax[1].set_yticks(ticks=[0,N])
ax[1].set_yticklabels(["$x_1$","$x_L$"])


im3=ax[2].imshow(True_testing - testing_prediction.T,aspect = "auto", cmap="jet", 
                  extent=[0,t_testing/lyapunov_time,0,N]
                  )
ax[2].set_ylabel("$x$-$\overline {x}$", fontsize = fontsize_)
ax[2].tick_params(axis='both', which='major', labelsize=labelsize_)
ax[2].set_yticks(ticks=[0,N])
ax[2].set_yticklabels(["$x_1$","$x_L$"])

cbar_ax = fig.add_axes([0.93, 0.07, 0.02, 0.915])
all_values = True_testing.copy()


im1.set_clim(np.min(all_values),np.max(all_values))
im2.set_clim(np.min(all_values),np.max(all_values))
im3.set_clim(np.min(all_values),np.max(all_values))

clb=fig.colorbar(im1, cax=cbar_ax)
clb.ax.tick_params(labelsize=labelsize_clb) 

plot_time = (t[int((t_transient+t_training)/dt_eval):int((t_transient+t_training+t_testing)/dt_eval)] - t[int((t_transient+t_training)/dt_eval)]) /lyapunov_time
prediction_horizon=plot_time[np.where(NRMSE> 0.3)[0][0]]

im4=ax[3].plot(plot_time,NRMSE,lw=2)
if lyapunov_time == 1:
    ax[3].set_xlabel("Time [MTU]", fontsize = fontsize_)
else:
    ax[3].set_xlabel("Time/$\lambda$", fontsize = fontsize_)

ax[3].axvline(x=prediction_horizon,color='k', linestyle='--',lw=2)


ax[3].set_ylabel("NRMSE", fontsize = fontsize_)
ax[3].tick_params(axis='both', which='major', labelsize=labelsize_)
ax[3].set_ylim(-0.1,2.1)

ax[0].text(-0.6,0.91*N,"(a)", fontsize=20)
ax[1].text(-0.6,0.91*N,"(b)", fontsize=20)
ax[2].text(-0.6,0.91*N,"(c)", fontsize=20)
ax[3].text(-0.6,1.88,"(d)", fontsize=20)

plt.show()


