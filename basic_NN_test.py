#================================================================
#
#   File name   : basic_NN_test.py
#   Author      : Josiah Tan
#   Created date: 23/06/2020
#   Description : tests a basic NN model
#
#================================================================

#================================================================

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import joblib
import pandas as pd
import cv2
from PIL import Image
import os
import h5py
import pyautogui

#importing Josiah's image processing module (I should really change the name utils...
import utils

#importing Josiah's dnn modules
from JTDeepNet import name, layers, activations, optimisers, losses
from JTDeepNet.model import JTDNN, Input
from JTDeepNet.random_utils import feature_norm, plot_decision_boundary, mini_batch_generator

#import Marco's/ Sharon's modules
import screenCapture

#import Pranav's module
from vector_converter import press_keys

# configuration for screen capture
width = 640
height = 360
#x_screen_corner = 1280 / 2 - width / 2
#y_screen_corner = 720 / 2 - height / 2
x_screen_corner = 0
y_screen_corner = 50
print_time = False
show_screen = False

# configuration for black and white
gray_dims = (25, 25)
gray_output_filename = "gray_datasets.h5"

# configuration for canny
use_canny = False

#configuration for RGB
RGB_dims = (416, 416)
RGB_output_filename = "RGB_datasets.h5"

# figure numbers
fig_num_cost = 2

#loadin
model_name = "basic_NN_DRC.plk"

basic_NN = joblib.load(model_name)

processimg_obj = utils.JTImageProcessing()

start = input("start? ")

while True:
    
    screen_capture = screenCapture.screen_record(x_screen_corner, y_screen_corner, width, height, print_time, show_screen)
    
    gray_img = screenCapture.RGB2Gray(screen_capture)
    
    resized_gray = processimg_obj.image_resize(gray_img, gray_dims)
    
    if use_canny:
        canned_img = screenCapture.Can(resized_gray)
        cv2.imshow('canned_image',canned_img )
        X_unravel = canned_img.reshape(-1, gray_dims[0] * gray_dims[1])
    else:
        cv2.imshow('resized_gray', resized_gray)
        X_unravel = resized_gray.reshape(-1, gray_dims[0] * gray_dims[1])
        
    resized_rgb = processimg_obj.image_resize(screen_capture, RGB_dims)
    

    
    
    X_T = X_unravel.T
    X_standard = X_T / 255
    X = X_standard
    
    AL = basic_NN.forward_prop(X)
    prediction = AL.T >= 0.5
    print(prediction.squeeze(), flush = True)
    
    press_keys(prediction.squeeze())

    if cv2.waitKey(25) & 0xFF == ord('q'): # for breaking out of the loop when q is pressed
        cv2.destroyAllWindows()
        break
# optionally put a loop here so that you start 
# listening again after the connection closes

"""
processimg_obj = utils.JTImageProcessing()

x, y, classes = processimg_obj.read_whole_dataset(key_x = "gray_x", key_y = "raw_y", key_classes = "classes", username = "Josiah", index = "all", dataset_file_name = gray_output_filename, mode = 'r')

X =  np.concatenate(x, axis = 0)
Y =  np.concatenate(y, axis = 0)

print(f"X.shape {X.shape}")
print(f"Y.shape {Y.shape}")
print(f"classes {classes}")

# reshapin for deep NN

X_unravel = X.reshape(-1, gray_dims[0] * gray_dims[1])
print(f"X_unravel.shape {X_unravel.shape}")

Y_T = Y.T
X_T = X_unravel.T

print(f"Y_T.shape {Y_T.shape}")
print(f"X_T.shape {X_T.shape}") 

#standardising the X values
X_standard = X_T / 255

X = X_standard
Y = Y_T

print(f"X.shape {X.shape}")
print(f"Y.shape {Y.shape}")

# buildin the DNN

file_name = 'cubic_model.plk'
basic_NN = JTDNN()
input = basic_NN.input(input_dims = (gray_dims[0] * gray_dims[1], None))
Z1 = layers.Linear(output_dims = (20, None), initialiser = "glorot", name = "linear")(input)
A1 = activations.Sigmoid(Z1, name = 'sigmoid')
Z2 = layers.Linear(output_dims = (10, None), initialiser = "glorot", name = "linear")(A1) 
A2 = activations.Sigmoid(Z2, name = 'sigmoid')
Z3 = layers.Linear(output_dims = (len(classes), None), initialiser = "glorot", name = "linear")(A2)
output = activations.Sigmoid(Z3, name = 'sigmoid')

#optimiser = optimisers.GradientDesc(learning_rate = 0.001)
optimiser = optimisers.Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon=1e-07)


loss = losses.BinaryCrossEntropy(basic_NN, store_cost = True, fig_num = fig_num_cost)

basic_NN.compile(input = input, output = output, lambd = 0.01, loss = loss, metrics = "Accuracy", optimiser = optimiser) # BGD stands for Batch Gradient Descent # BGD stands for Batch Gradient Descent



for epoch in range(num_epoches):
    mini_batch_num = 1
    for mini_batch_X, mini_batch_Y in mini_batch_generator(X, Y, mini_batch_size):
        
        AL = basic_NN.forward_prop(mini_batch_X)
        
        cost = basic_NN.compute_cost(mini_batch_Y)
        
        print('epoch %d accuracy after iteration %d: %4.2f' % (epoch, mini_batch_num, np.mean((AL >= 0.5) == mini_batch_Y) * 100))
        basic_NN.back_prop(mini_batch_Y)
        basic_NN.update_weights()
        mini_batch_num +=1


#plot_decision_boundary(X, Y, basic_NN, fig_num_dec)

basic_NN.plot_cost(title = "Cost per Iteration", xlabel = "Number of iterations", ylabel = "Cost")

plt.show()

joblib.dump(basic_NN, model_name)
"""




