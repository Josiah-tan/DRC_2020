#================================================================
#
#   File name   : basic_NN_train.py
#   Author      : Josiah Tan
#   Created date: 23/06/2020
#   Description : trains a basic NN model
#
#================================================================

#================================================================
from sklearn.metrics import f1_score

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import joblib
import pandas as pd
import cv2
from PIL import Image
import os
import h5py

#import Marco's/ Sharon's modules
import screenCapture

#importing Josiah's image processing module (I should really change the name utils...
import utils

#importing Josiah's dnn modules
from JTDeepNet import name, layers, activations, optimisers, losses
from JTDeepNet.model import JTDNN, Input
from JTDeepNet.random_utils import feature_norm, plot_decision_boundary, mini_batch_generator, sim_data_inversion_Y
from deep_net_framework import MulticlassDnn
# configuration for black and white
gray_dims = (25, 25) # 100, 100
gray_output_filename = "gray_datasets.h5"

#configuration for RGB
RGB_dims = (416, 416)
RGB_output_filename = "RGB_datasets.h5"

# figure numbers
fig_num_cost = 2
fig_num_dec = 1

#training
mini_batch_size = 1024
num_epoches = 2000 # 3000

#loadin
load_model_name = "basic_NN_DRC.plk"
load_model = True

#savin
save_model_name = "basic_NN_DRC.plk"
save_model = True


processimg_obj = utils.JTImageProcessing()

x, y, classes = processimg_obj.read_whole_dataset(key_x = "gray_x", key_y = "raw_y", key_classes = "classes", username = "Josiah", index = "all", dataset_file_name = gray_output_filename, mode = 'r')

raw_X =  np.concatenate(x, axis = 0)
raw_Y =  np.concatenate(y, axis = 0)

print(f"raw_X.shape {raw_X.shape}")
print(f"raw_Y.shape {raw_Y.shape}")
print(f"classes {classes}")

X_resized = []
for img in range(raw_X.shape[0]):
    X_resized.append(processimg_obj.image_resize(raw_X[img], gray_dims)) # resizes all the images

raw_X = np.array(X_resized)
print(f"X_resized.shape {raw_X.shape}") # not a mistake



inv_X, inv_Y, count = sim_data_inversion_Y(raw_X, raw_Y)

print(f"inv_X.shape { inv_X.shape}")
print(f"inv_Y.shape { inv_Y.shape}")

X = np.concatenate((raw_X, inv_X), axis = 0)
Y = np.concatenate((raw_Y, inv_Y), axis = 0)

print(f"X.shape {X.shape}")
print(f"Y.shape {Y.shape}")

"""testing if the number of right and left are equal"""
"""
right = np.array([1,0,1,0])
count_right = 0

left = np.array([0,1,1,0])
count_left = 0
for ind in range(Y.shape[0]):
    count_right += np.array_equal(Y[ind], right)
    count_left  += np.array_equal(Y[ind], left)

print(f"count_left {count_right}")
print(f"count_right {count_right}")
"""
    
"""
# testing if inversion works
ind = 4
plt.subplot(1,2,1)
plt.imshow(screenCapture.Can(raw_X[count[ind]]), 'gray')
plt.subplot(1,2,2)
plt.imshow(screenCapture.Can(inv_X[ind]), 'gray')
plt.show()
print(raw_Y[count[ind]])
print(inv_Y[ind])
exit()
"""

"""process all images with Canny"""
"""
for img in range(X.shape[0]):
    plt.subplot(1,2,1)
    plt.imshow(X[img])
    X[img] = screenCapture.Can(X[img]) # converting to Canny
    plt.subplot(1,2,2)
    plt.imshow(X[img])
    plt.show()
"""


""" is this the kind of image I want?"""
for i in range(8):
    plt.subplot(2,4,i + 1)
    plt.imshow(X[i+1000])
plt.show()
# reshapin for deep NN

X_unravel = X.reshape(-1, gray_dims[0] * gray_dims[1])
print(f"X_unravel.shape {X_unravel.shape}")

Y_T = Y.T
X_T = X_unravel.T

print(f"Y_T.shape {Y_T.shape}")
print(f"X_T.shape {X_T.shape}") 

#standardising the X values
X_standard = X_T / 255



"""
# check whether transpose does anythin dodgy
X_reverted = X_unravel.reshape(X_unravel.shape[0], gray_dims[0], gray_dims[1])
print(X_reverted.shape)
print(X.shape)
print(X_reverted == X)
assert (X_reverted == X).all()

plt.imshow(processimg_obj.image_resize(X_reverted[178,:,:], (25,25)), "gray")
print(Y[178,:])
plt.show()
exit()
"""
X = X_standard
Y = Y_T

print(f"X.shape {X.shape}")
print(f"Y.shape {Y.shape}")

"""for train/val/test"""

insect_dnn = MulticlassDnn()

ratios = [92,4,4]

train_x_orig, train_y, val_x_orig, val_y, test_x_orig, test_y = insect_dnn.train_val_test(ratios, X.T, Y)
train_x = train_x_orig.T
val_x = val_x_orig.T
test_x = test_x_orig.T

print(f"train_x {train_x.shape}")
print(f"val_x {val_x.shape}")
print(f"test_x {test_x.shape}")
print(f"train_y {train_y.shape}")
print(f"val_y {val_y.shape}")
print(f"test_y {test_y.shape}")
#exit()
# x_orig -- contains all x values with x.shape = (m, -1) (the number of examples, matrix containing features)
# y_orig -- contains all y values with y.shape = (1, m) 

# buildin the DNN

if not(load_model):
    basic_NN = JTDNN()
    input = basic_NN.input(input_dims = (gray_dims[0] * gray_dims[1], None))
    Z1 = layers.Linear(output_dims = (80, None), initialiser = "glorot", name = "linear")(input) #10
    A1 = activations.ReLu(Z1, name = 'relu')
    Z2 = layers.Linear(output_dims = (40, None), initialiser = "glorot", name = "linear")(A1) #10
    A2 = activations.ReLu(Z2, name = 'relu')
    Z3 = layers.Linear(output_dims = (20, None), initialiser = "glorot", name = "linear")(A2) # 5
    A3 = activations.ReLu(Z3, name = 'relu')
    Z4 = layers.Linear(output_dims = (len(classes), None), initialiser = "glorot", name = "linear")(A3)
    output = activations.Sigmoid(Z4, name = 'sigmoid')
    

if load_model:
    basic_NN = joblib.load(load_model_name)

"""compilin stuff here"""    
#optimiser = optimisers.GradientDesc(learning_rate = 0.001)
optimiser = optimisers.Adam(learning_rate = 0.0001, beta_1 = 0.98, beta_2 = 0.999, epsilon=1e-07) # 0.0001 # beta_1 0.9


loss = losses.BinaryCrossEntropy(basic_NN, store_cost = True, fig_num = fig_num_cost)

# different requirements essentially for loading and making a new model
if load_model:
    basic_NN.compile(lambd = 0.01, loss = loss, metrics = "Accuracy", optimiser = optimiser) # BGD stands for Batch Gradient Descent # BGD stands for Batch Gradient Descent
else:
    basic_NN.compile(input = input , output = output, lambd = 0.01, loss = loss, metrics = "Accuracy", optimiser = optimiser) # BGD stands for Batch Gradient Descent # BGD stands for Batch Gradient Descent

best_val_mean_f1 = 0 # so that we can save the model with highest f1 score for validation
val_costs = [] # store the costs for validation

for epoch in range(num_epoches):
    mini_batch_num = 1
    for mini_batch_X, mini_batch_Y in mini_batch_generator(train_x, train_y, mini_batch_size):
        
        AL = basic_NN.forward_prop(mini_batch_X)
        
        cost = basic_NN.compute_cost(mini_batch_Y)
        
        #print('epoch %d accuracy after iteration %d: %4.2f' % (epoch, mini_batch_num, np.mean((AL >= 0.5) == mini_batch_Y) * 100))
        predict = AL >= 0.5
        d = f1_score(mini_batch_Y[0], predict[0])
        a = f1_score(mini_batch_Y[1], predict[1])
        w = f1_score(mini_batch_Y[2], predict[2])
        s = f1_score(mini_batch_Y[3], predict[3])
        print('epoch %d training f1_score after iteration %d: d: %4.2f a: %4.2f  w: %4.2f  s: %4.2f ave = %4.2f' % (epoch, mini_batch_num, d, a, w, s, (d + a + w + s)/4))
        print(np.mean( predict == mini_batch_Y, axis = 1) * 100)
        basic_NN.back_prop(mini_batch_Y)
        basic_NN.update_weights()
        mini_batch_num +=1
    
    """performing cross validation check"""
    print("VALIDATION") # note to self, toss all these evaluation stuff in a function pleasssee this is so messy
    AL = basic_NN.forward_prop(val_x)
    
    cost = basic_NN.compute_cost(val_y)
    val_costs.append(cost) # appending for graphing later
    #print('epoch %d accuracy after iteration %d: %4.2f' % (epoch, mini_batch_num, np.mean((AL >= 0.5) == mini_batch_Y) * 100))
    predict = AL >= 0.5
    d = f1_score(val_y[0], predict[0])
    a = f1_score(val_y[1], predict[1])
    w = f1_score(val_y[2], predict[2])
    s = f1_score(val_y[3], predict[3])
    mean_f1 = (d + a + w + s) / 4
    print('epoch %d validation f1_score after iteration %d: d: %4.2f a: %4.2f  w: %4.2f  s: %4.2f ave = %4.2f' % (epoch, mini_batch_num, d, a, w, s, mean_f1))
    print(np.mean( predict == val_y, axis = 1) * 100)
    
    
    if mean_f1 > best_val_mean_f1:
        best_val_mean_f1 = mean_f1
        print(f"model improved to {best_val_mean_f1}")
        if save_model:
            joblib.dump(basic_NN, save_model_name)
            print("model has been saved")
    else:
        print(f"model did not improve since {best_val_mean_f1}")
    print("END_VALIDATION")
"""
for itera in range(1000):
    AL = basic_NN.forward_prop(X)
    if itera % 100 == 0:
        loss = basic_NN.compute_cost(Y)
        print(loss)
        #print('accuracy after iteration %d: %4.2f' % itera, np.mean((AL >= 0.5) == Y) * 100)
    basic_NN.back_prop(Y)
    
    basic_NN.update_weights()
basic_NN.plot_cost(title = "Cost per Iteration", xlabel = "Number of number of iterations (10000s)", ylabel = "Cost")
"""
#plot_decision_boundary(X, Y, basic_NN, fig_num_dec)

basic_NN.plot_cost(title = "Cost per Iteration", xlabel = "Number of iterations", ylabel = "Cost")

plt.figure(1)
plt.plot(val_costs)
plt.xlabel("Number of epoches")
plt.ylabel("Costs")
plt.title("Cost per epoche")

plt.show()


"""error analysis"""
for img_num in range(val_x.shape[-1]):
    AL = basic_NN.forward_prop(val_x[:,img_num])
    predict = AL.T >= 0.5
    if not np.array_equal(predict, val_y[:, img_num]):
        plt.imsave(os.path.join(r'C:\Users\josia\Desktop\Josiah_Folder\UNI\Semester_1\PEP1\robotics_club\YOLOv3_tiny\error_analysis_val', f"{''.join([str(int(b)) for b in predict.squeeze().tolist()])} {img_num}.jpg"), val_x[:,img_num] * 255)





