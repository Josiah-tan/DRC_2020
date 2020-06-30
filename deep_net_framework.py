#! python3
#Josiah's personal deep net framework
#Author Josiah Tan
#https://github.com/lalitkpal/ImageClassification/blob/master/CatsVsDogs/VGG16_extend.py

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

from PIL import Image
import io
import scipy
from scipy import ndimage

from test_cases import *
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward, softmax, softmax_backward, gradients_to_vector, vector_to_dictionary, dictionary_to_vector
from coding_neural_network_from_scratch import (initialize_parameters,
                                                L_model_forward,
                                                L_model_backward,
                                                compute_cost,
                                                gradient_check,
                                                forward_prop_cost)

import joblib

#jtdecorators:
import functools
import time
from jtdecor import timer

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

dataset_path = 'datasets.h5'

program = 'decision_boundary_2D'

"""
TODO:
0-  Perform gradient checking
1-    Add minibatches as a feature    
DONEEE 2-  make the __name__ == "__main__":
        main()
        #allows use as a module without actually running the code
3-  Add dropout
4-  Add batch Norm (good stuff)
"""

class Dnn:
    def __init__(self):
        self.keep_results = False
        self.results = {}
        #self.layers_dims
        #self.reset_results
        self.costs = []

    def load_results(self, layer_dims):
        """
        load_results: needs joblib to be imported, the file names end with .plk
        
        Arguments:
        layer_dims -- a list containing the dimensions of the layers
        
        Returns -- dictionary containing the parameters/ weights of the model
        """
        file_name = self.layer_dims2file_name(layer_dims, extension = "_results_logs.plk")
        print(f"loading {file_name}")
        return joblib.load(file_name)    
    def save_results(self, layer_dims, reset = True):
        """
        save_hyperparameters: needs joblib to be imported, the file names have notation: x_y_z_w_hyperparameters, records results in current directory
        
        Arguments:
            layer_dims -- a list containing the dimensions of the layers
            
            self.results -- dictionary containing the results of the model 
                            e.g {accuracy_val : 0.0008
                                 accuracy_test: 100000
                                 confusion_val : 
                                 confusion_test :
                                 costs:
                                }
            creates a file with contents:
                    e.g: = {"improvement_1" : {accuracy_val: ....
                                        }
                            "improvement_2" : {accuracy_val: ...
                                        }
                           }
            reset -- if reset is true, overwrite file (if it exists) then start from "improvement_1" again               
            
        returns none
        """
        
        file_name = self.layer_dims2file_name(layer_dims, extension = "_results_logs.plk")
        
        if not(reset):
            print("if not(reset)")
            if os.path.isfile(file_name):
                print("placeholder")
                loaded_results = self.load_results(layer_dims)
                print(f"loaded_hyperparameters {loaded_results}")
                print(f"len(loaded_hyperparameters) {len(loaded_results)}")
                num_improvements = len(loaded_results) # how many improvements have been made to the model
                print(num_improvements)
                loaded_results[f"improvement_{num_improvements + 1}"] = self.results
                print(loaded_results)
                joblib.dump(loaded_results, file_name)
                print("Saving results (1)")            
            else:
                print("Saving results (2)")
                joblib.dump({"improvement_1":self.results}, file_name)
           
        else:
            print("Saving results (3)")
            #print(f"file_name {file_name}")
            #print(self.results)
            joblib.dump({"improvement_1":self.results}, file_name)
        
    def call_destructor(self):
        #class destructor imitator
        if self.keep_results:
            self.save_results(self.layers_dims, self.reset_results)
    
    def error_analysis(self, y_hat, x, y, classes, num_px, error_directory):
        """
        error_analysis, compares the ground truth with predicted values, and outputs a folder with files containing mislabelled images
        
        parameters -- x: normalised x input values, shape: (n, m) (UNTESTED)  or (m, num_px, num_px, 3) (TESTED)
                   -- y: y ground truth values, shape: (1, m)
                   -- y_hat: predicted y values, shape: (1, m)
                   -- classes, list containing the classes encoded in "utf-8"
                   -- folder_name, contains the name of the folder where the images will be put in
        """
        
        
        #added this in without testing for the DNN case, only the CNN since I did not want to convert the shape twice:
        if len(x.shape) != 4:
            x = x.T #(m,n)
            x = x.reshape(x.shape[0], num_px, num_px, 3) # (m, num_px, num_px, 3)
        for example in range(x.shape[0]):
            
            #print(example)
            #print(x[example])
            #print(x[example].shape)
            #print(x.shape)
            if y_hat[0, example] != y[0, example]:
            
                y_example = classes[y[0, example]].decode("utf-8") #classes are encoded in utf-8
                
                y_hat_example = classes[y_hat[0, example]].decode("utf-8")
                
                #classes[train_y[0,index]].decode("utf-8")
                im = Image.fromarray((x[example]*255).astype(np.uint8)) # image.fromarray() only accepts unit8 and x[example] was normalised [0,255] -> [0,1]
                if not(os.path.isdir(error_directory)):
                    os.mkdir(error_directory)
                im.save(os.path.join(error_directory, f"example_{example + 1}_true_label_{y_example}_pred_{y_hat_example}.jpg")) #put incorrectly labelled images into the error_directory
                
    
    def layer_dims2file_name(self, layer_dims, extension):
        """
        layer_dims2file_name: converts a layer_dims into a file name [x,y,z,w] + extension -> x_y_z_wextension e.g: x_y_z_w_weights.plk (extension = _weights.plk)
        """
        layer_dims = [str(layers) for layers in layer_dims] # convert layer sizes into strings
        
        file_name_head = "_".join(layer_dims) # join all these layers into the form "x_y_z_w", for example: "100_50_20_5"
        #print(file_name_head)
        
        file_name = file_name_head + extension
        #print(file_name)
        
        return file_name

    def load_hyperparameters(self, layer_dims):
        """
        load_hyperparameters: needs joblib to be imported, the file names end with .plk
        
        Arguments:
        layer_dims -- a list containing the dimensions of the layers
        
        Returns -- dictionary containing the parameters/ weights of the model
        """
        file_name = self.layer_dims2file_name(layer_dims, extension = "_hyperparameter_logs.plk")
        print(f"loading {file_name}")
        return joblib.load(file_name)    
    def save_hyperparameters(self, layer_dims, hyperparameters, reset = True):
        """
        save_hyperparameters: needs joblib to be imported, the file names have notation: x_y_z_w_hyperparameters, records hyperparameters in current directory
        
        Arguments:
            layer_dims -- a list containing the dimensions of the layers
            
            hyperparameters -- dictionary containing the hyperparameters of the model 
                            e.g {learning_rate : 0.0008
                                 num_iterations: 100000
                                }
            creates a file with contents:
                    e.g: = {"improvement_1" : {learning_rate : 0.0008
                                         num_iterations: 100000
                                        }
                            "improvement_2" : {learning_rate : 0.0001
                                         num_iterations: 1000000
                                        }
                           }
            reset -- if reset is true, overwrite file (if it exists) then start from "improvement_1" again               
            
        returns none
        """
        
        file_name = self.layer_dims2file_name(layer_dims, extension = "_hyperparameter_logs.plk")
        
        if not(reset):
            try:
                #print("placeholder")
                loaded_hyperparameters = self.load_hyperparameters(layer_dims)
                #print(f"loaded_hyperparameters {loaded_hyperparameters}")
                #print(f"len(loaded_hyperparameters) {len(loaded_hyperparameters)}")
                num_improvements = len(loaded_hyperparameters) # how many improvements have been made to the model
                #print(num_improvements)
                loaded_hyperparameters[f"improvement_{num_improvements + 1}"] = hyperparameters
                #print(loaded_hyperparameters)
                joblib.dump(loaded_hyperparameters, file_name)
                print("Saving hyperparameters (1)")            
            except:
                print("Saving hyperparameters (2)")
                joblib.dump({"improvement_1":hyperparameters}, file_name)
           
        else:
            print("Saving hyperparameters (3)")
            joblib.dump({"improvement_1":hyperparameters}, file_name)
    
    def load_weights(self, layer_dims):
        """
        load_weights: needs joblib to be imported, the file names end with .plk, can replace the function initialise_parameters deep
        
        Arguments:
        layer_dims -- a list containing the dimensions of the layers
        
        Returns -- dictionary containing the parameters/ weights of the model
        """
        file_name = self.layer_dims2file_name(layer_dims, extension = "_weights.plk")
        
        return joblib.load(file_name)
        
    
    def save_weights(self, layer_dims, weights):
        """
        save_weights: needs joblib to be imported, the file names have notation: x_y_z_w_weights.plk, saves weights in current directory
        
        Arugments:
        layerdims -- a list containing the dimensions of the layers
        weights -- dictionary containing the parameters/ weights of the model
        
        Returns none
        """
        #print(f"layer_dims {layer_dims}")
        #print(f"weights {weights}")
        
        file_name = self.layer_dims2file_name(layer_dims, extension = "_weights.plk")
        
        try: 
            joblib.dump(weights, file_name)
            print(f"storing weights in {file_name}")
        except:
            print("Hey M8 there is an error!!!! you gotta import joblib")
    
    def initialize_parameters_deep(self, layer_dims):
        """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network
        
        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """
        
        np.random.seed(1)
        parameters = {}
        L = len(layer_dims)            # number of layers in the network

        for l in range(1, L):
            ### START CODE HERE ### (≈ 2 lines of code)
            parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) * np.sqrt(2/layer_dims[l-1]) # Xavier initialisation
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
            ### END CODE HERE ###
            #print(parameters['W' + str(l)].shape)
            #print(parameters['b' + str(l)].shape)
            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

            
        return parameters


    def linear_forward(self, A, W, b):
        """
        Implement the linear part of a layer's forward propagation.

        Arguments:
        A -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
        Z -- the input of the activation function, also called pre-activation parameter 
        cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
        """
        
        ### START CODE HERE ### (≈ 1 line of code)
        Z = np.dot(W,A) +b
        ### END CODE HERE ###
        
        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)
        
        return Z, cache


    def linear_activation_forward(self, A_prev, W, b, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        A -- the output of the activation function, also called the post-activation value 
        cache -- a python tuple containing "linear_cache" and "activation_cache";
                 stored for computing the backward pass efficiently
        """
        
        if activation == "sigmoid":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            ### START CODE HERE ### (≈ 2 lines of code)
            Z, linear_cache = self.linear_forward(A_prev,W,b)
            A, activation_cache = sigmoid(Z)
            ### END CODE HERE ###
        
        elif activation == "relu":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            ### START CODE HERE ### (≈ 2 lines of code)
            Z, linear_cache = self.linear_forward(A_prev,W,b)
            A, activation_cache = relu(Z)
            ### END CODE HERE ###
        
        elif activation == "softmax":
            
            Z, linear_cache = self.linear_forward(A_prev,W,b)
            A, activation_cache = softmax(Z)
        
        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache


    def L_model_forward(self, X, parameters):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
        
        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()
        
        Returns:
        AL -- last post-activation value
        caches -- list of caches containing:
                    every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
        """

        caches = []
        A = X
        L = len(parameters) // 2                  # number of layers in the neural network
        
        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            A_prev = A 
            ### START CODE HERE ### (≈ 2 lines of code)
            A, cache = self.linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")

            caches.append(cache)
            ### END CODE HERE ###
        
        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        ### START CODE HERE ### (≈ 2 lines of code)
        AL, cache = self.linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
        caches.append(cache)
        ### END CODE HERE ###
        
        assert(AL.shape == (1,X.shape[1]))
                
        return AL, caches

    def compute_cost(self, AL, Y):
        """
        Implement the cost function defined by equation (7).

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """
        assert(AL.shape == Y.shape)
        m = Y.shape[1]
        

        # Compute loss from aL and y.
        ### START CODE HERE ### (≈ 1 lines of code

        cost = -1/m*np.sum(np.multiply(Y,np.log(AL))+np.multiply(1-Y,np.log(1-AL)))
        #print (cost)
        ### END CODE HERE ###
        
        cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert(cost.shape == ())
        
        return cost

    def compute_cost_with_regularization(self, AL, Y, parameters, lambd, layer_dims):
        """
        Implement the cost function with L2 regularization.
        
        Arguments:
        AL-- post-activation, output of forward propagation, of shape (output size, number of examples)
        Y -- "true" labels vector, of shape (output size, number of examples)
        parameters -- python dictionary containing parameters of the model
        
        Returns:
        cost - value of the regularized loss function (formula (2))
        """
        assert(Y.shape == AL.shape)
        
        m = Y.shape[1]
        
        L = len(layer_dims)            # number of layers in the network

        sum_w_squared = 0 # sum of w squared
        
        for l in range(1, L):
            sum_w_squared += np.sum(np.square(parameters['W' + str(l)])) 

        
        #cross_entropy_cost = np.squeeze(-1/m*np.sum(np.multiply(Y,np.log(AL))+np.multiply(1-Y,np.log(1-AL)))) # This gives you the cross-entropy part of the cost, do not use the self.compute_cost(), gives different results for some reason
        
        losses = np.multiply(Y,np.log(AL))+np.multiply(1-Y,np.log(1-AL))
        """This checks if any elements are NAN, if they are, it also checks if y^i_j == al^i_j since 0log0 = 0 not nan"""
        if np.isnan(losses).any():
            #print(f'losses {losses}')
            #print(f'np.isnan(losses) \n{np.isnan(losses)}')
            #print(f'Y == AL \n{Y == AL}')
            #print(f'(Y == AL) * np.isnan(losses) \n{(Y == AL) * np.isnan(losses)}')
        
            losses[(Y == AL) * np.isnan(losses)] = 0
        
            #print(f'new losses {losses}')
        
        cross_entropy_cost = np.squeeze(-1/m * np.sum(losses))
        
            
        ### START CODE HERE ### (approx. 1 line)
        L2_regularization_cost = lambd * (sum_w_squared) / (2 * m)
        ### END CODER HERE ###
        
        cost = cross_entropy_cost + L2_regularization_cost
        
        return cost

    def linear_backward(self, dZ, cache): 
        """
        Implement the linear portion of backward propagation for a single layer (layer l)

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        #print(len(cache))
        A_prev, W, b = cache
        m = A_prev.shape[1]

        ### START CODE HERE ### (≈ 3 lines of code)
        dW = 1/m * np.dot(dZ, A_prev.T)
        db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
        dA_prev = np.dot(W.T, dZ)
        ### END CODE HERE ###
        
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        
        return dA_prev, dW, db


    def linear_backward_with_regularisation(self, dZ, cache, lambd): 
        """
        Implement the linear portion of backward propagation for a single layer (layer l)

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        #print(len(cache))
        A_prev, W, b = cache
        m = A_prev.shape[1]

        ### START CODE HERE ### (≈ 3 lines of code)
        dW = 1/m * np.dot(dZ, A_prev.T)  + lambd/m * W
        db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
        dA_prev = np.dot(W.T, dZ)
        ### END CODE HERE ###
        
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        
        return dA_prev, dW, db
        

    def linear_activation_backward(self, dA, cache, activation):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.
        
        Arguments:
        dA -- post-activation gradient for current layer l 
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
        
        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        linear_cache, activation_cache = cache
        
        if activation == "relu":
            ### START CODE HERE ### (≈ 2 lines of code)
            dZ = relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
            ### END CODE HERE ###
            
        elif activation == "sigmoid":
            ### START CODE HERE ### (≈ 2 lines of code)
            dZ = sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
            ### END CODE HERE ###
            
        elif activation == "softmax":
            
            dZ = softmax_backward(dA, activation_cache) #not implemented yet
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        
        return dA_prev, dW, db

    def linear_activation_backward_with_regularisation(self, dA, cache, activation, lambd):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.
        
        Arguments:
        dA -- post-activation gradient for current layer l 
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
        
        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        linear_cache, activation_cache = cache
        
        if activation == "relu":
            ### START CODE HERE ### (≈ 2 lines of code)
            dZ = relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward_with_regularisation(dZ, linear_cache, lambd)
            ### END CODE HERE ###
            
        elif activation == "sigmoid":
            ### START CODE HERE ### (≈ 2 lines of code)
            dZ = sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward_with_regularisation(dZ, linear_cache, lambd)
            ### END CODE HERE ###
            
        elif activation == "softmax": #not implemented yet
            
            dZ = softmax_backward(dA, activation_cache) 
            dA_prev, dW, db = self.linear_backward_with_regularisation(dZ, linear_cache, lambd)
        
        return dA_prev, dW, db
        

    def L_model_backward(self, AL, Y, caches, lambd):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
        
        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                    the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
        
        Returns:
        grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ... 
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ... 
        """
        grads = {}
        L = len(caches) # the number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
        
        # Initializing the backpropagation
        ### START CODE HERE ### (1 line of code)
        #dAL = -(np.divide(Y,AL) - np.divide(1-Y,1-AL))
        ### END CODE HERE ###
        
        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
        ### START CODE HERE ### (approx. 2 lines)
        #current_cache = caches[-1]
        #grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward_with_regularisation(dAL, current_cache, activation = "sigmoid", lambd = lambd)
        ### END CODE HERE ###
        # Loop from l=L-2 to l=0
        dZ = AL - Y
        ### END CODE HERE ###
        
        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
        ### START CODE HERE ### (approx. 2 lines)
        current_cache = caches[-1]
        #grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, activation = "softmax")
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_backward_with_regularisation(dZ, current_cache[0], lambd)
        
        
        """
        try:
            assert np.isclose(grads["dA" + str(L - 1)], hi[0]).all()
        except:
            print(f'grads["dA" + str(L-1)] {grads["dA" + str(L-1)]}')
            print(f'hi[0] {hi[0]}')
        try:
            assert np.isclose(grads["dW" + str(L)], hi[1]).all() 
        except:
            pass
        try:
            assert np.isclose(grads["db" + str(L)], hi[2]).all()
        except:
            pass
        """
        
        for l in reversed(range(L-1)):
            # lth layer: (RELU -> LINEAR) gradients.
            # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
            ### START CODE HERE ### (approx. 5 lines)
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward_with_regularisation(grads["dA" + str(l+1)], current_cache, activation = "relu", lambd = lambd)
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
            ### END CODE HERE ###

        return grads


    def update_parameters(self, parameters, grads, learning_rate):
        """
        Update parameters using gradient descent
        
        Arguments:
        parameters -- python dictionary containing your parameters 
        grads -- python dictionary containing your gradients, output of L_model_backward
        
        Returns:
        parameters -- python dictionary containing your updated parameters 
                      parameters["W" + str(l)] = ... 
                      parameters["b" + str(l)] = ...
        """
        
        L = len(parameters) // 2 # number of layers in the neural network

        # Update rule for each parameter. Use a for loop.
        ### START CODE HERE ### (≈ 3 lines of code)
        for l in range(L):
            parameters["W" + str(l+1)] -= learning_rate * grads["dW" +str(l+1)]
            parameters["b" + str(l+1)] -= learning_rate * grads["db" +str(l+1)]
        ### END CODE HERE ###
        return parameters
        

    def disp_cost(self, i, cost):
        if i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    def L_layer_model(self, X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False, lambd = 0, parameters = None, show_plot = False):#lr was 0.009
        """
        Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
        
        Arguments:
        X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
        learning_rate -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop
        print_cost -- if True, it prints the cost every 100 steps
        
        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """

        np.random.seed(1)
        #costs = []                         # keep track of cost
        
        # Parameters initialization. (≈ 1 line of code)
        ### START CODE HERE ###
        if parameters is None:
            parameters = self.initialize_parameters_deep(layers_dims) #alg
        ### END CODE HERE ###
        
        # Loop (gradient descent)
        for i in range(0, num_iterations):

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            ### START CODE HERE ### (≈ 1 line of code)
            AL, caches = self.L_model_forward(X, parameters)
            ### END CODE HERE ###
            
            # Compute cost.
            ### START CODE HERE ### (≈ 1 line of code)
            #cost = self.compute_cost(AL, Y)
            cost = self.compute_cost_with_regularization(AL, Y, parameters, lambd, layers_dims)
            ### END CODE HERE ###
        
            # Backward propagation.
            ### START CODE HERE ### (≈ 1 line of code)
            grads = self.L_model_backward(AL, Y, caches, lambd)
            ### END CODE HERE ###
     
            # Update parameters.
            ### START CODE HERE ### (≈ 1 line of code)
            parameters = self.update_parameters(parameters, grads, learning_rate)
            ### END CODE HERE ###
                    
            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
            #if print_cost and i % 100 == 0:
            self.costs.append(cost)
                
            """if print_cost:
                self.disp_cost(i, cost)"""
                
        # plot the cost
        if show_plot:
            plt.plot(np.squeeze(self.costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per hundreds)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()
            
        return parameters
    
    
    def predict(self, X, y, parameters):
        """
        This function is used to predict the results of a  L-layer neural network.
        
        Arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model
        
        Returns:
        p -- predictions for the given dataset X
        """
        
        m = X.shape[1]
        n = len(parameters) // 2 # number of layers in the neural network
        p = np.zeros((1,m))
        
        # Forward propagation
        probas, caches = self.L_model_forward(X, parameters)

        
        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
        
        #print results
        #print ("predictions: " + str(p))
        #print ("true labels: " + str(y))
        
        p = probas >= 0.5
        accuracy = np.mean(labels == Y)
        print(f"Accuracy: {accuracy}")
            
        return p
    
    
    def train_val_test(self, ratios, x_orig, y_orig):
        
        # splits x_orig into training, validation and test sets
        # parameters: 
        # ratios -- list of ratios in the form [train, val, test]
        # x_orig -- contains all x values with x.shape = (m, -1) (the number of examples, matrix containing features)
        # y_orig -- contains all y values with y.shape = (1, m) 
        
        # returns:
        #train_x_orig, train_y, val_x_orig, val_y, test_x_orig, test_y
        
        m = x_orig.shape[0]
        
        ratio_array = np.array(ratios)
        
        ratio_array = np.floor(ratio_array / np.sum(ratio_array) * m); # divide each element by the sum of all elements then multiply by m, then round all down to nearest integer: 1.9 -> 1
        
        ratio_array[0] += m - np.sum(ratio_array); # add leftovers onto the first element- leftovers caused by floor function
        
        ratio_array = ratio_array.astype(int)
        
        perms = np.random.permutation(m) # rearrange indexes of x_orig
        
        index_range = [perms[0:ratio_array[0]], perms[ratio_array[0]: (ratio_array[0]+ratio_array[1])], perms[(ratio_array[0]+ratio_array[1]):m]] # organise indexes of x_orig
        #print(index_ranges)
        
        train_x_orig = x_orig[index_range[0]]
        
        train_y = y_orig[:,index_range[0]]
        
        #print(train_x_orig.shape)
        
        #print(train_y.shape)

        val_x_orig = x_orig[index_range[1]]
        
        val_y = y_orig[:, index_range[1]]

        #print(val_x_orig.shape)
        
        #print(val_y.shape)
        
        test_x_orig = x_orig[index_range[2]]
        
        test_y = y_orig[:, index_range[2]]
        
        #print(test_x_orig.shape)
        
        #print(test_y.shape)

        return train_x_orig, train_y, val_x_orig, val_y, test_x_orig, test_y
            

class MulticlassDnn(Dnn):
    def __init__(self):
        super().__init__()

    def print_image(image):
        #parameters: takes in a list of np.arrays or a dictionary containing key with array_name and the value of array itself
        #loads the images onto a figure
        #returns none
        print(type(image))
        if type(image)==list:
            #print (image[0].shape)
            fig = plt.figure(figsize = (8,8))
            for i in range(0,len(image)):
                fig.add_subplot(2,4,1 +i)
                plt.imshow(image[i])
            plt.show()
        elif type(image) == dict:
            fig = plt.figure(figsize = (8,8))
            i = 1
            for key, val in image.items():
                fig.add_subplot(2,4,i)
                i+=1
                plt.imshow(val)
                plt.xlabel(key)
            plt.show()
    
    #test if x_orig actually matches up with their respective classes
    def test_match(classes, x_orig, y_orig):
        #graphs images with class labels
        

        my_dict = {}
        perm = np.random.permutation(x_orig.shape[0])
        perm = np.random.permutation(perm)
        for  i in perm:
            #print(i)
            
            my_dict[classes[np.squeeze(y_orig[:,i])].decode('utf-8')] = x_orig[i]


        self.print_image(my_dict)
    
    
    def test_cost (self): # archived
        #may not even work in the future....
        AL = np.abs(np.random.randn(5,train_y.shape[1]))
    
        AL = AL/(np.max(AL)+10**(-5))
    
        insect_dnn.compute_cost(AL,train_y)
    
    
    def convert_y_to_binary_matrix(self, AL, Y):
        """
        Convert Y to a binary array- also known as one-hot encoding
        
        Arguments:
        Y -- true "label" vector with shape (1, number of examples (m))
        
        returns:
        array with zeroes of dimension(number of classes, m training examples)
        """
        
        Y = Y + np.zeros(AL.shape)
        class_index = np.arange(AL.shape[0]).reshape(AL.shape[0],1)
        Y = Y == class_index
        return Y
        
    def compute_cost(self, AL, Y): #overrides the function in base class

        """
        Implement the cost function.

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (number of classes, number of examples)
        Y -- true "label" vector with shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """
        
        #print(Y[:, 0:5])        
        """convert Y to an array with zeroes of dimension(number of classes, m training examples) """
        Y = self.convert_y_to_binary_matrix(AL, Y)
        #print(Y[:,0:5])
        return super().compute_cost(AL, Y)
        
    def compute_cost_with_regularization(self, AL, Y, parameters, lambd, layer_dims): #overrides the function in base class

        """
        Implement the cost function.

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (number of classes, number of examples)
        Y -- true "label" vector with shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """
        
        #print(Y[:, 0:5])        
        """convert Y to an array with zeroes of dimension(number of classes, m training examples) """
        Y = self.convert_y_to_binary_matrix(AL, Y)
        #print(Y[:,0:5])
        #return super().compute_cost(AL, Y)
        return super().compute_cost_with_regularization(AL, Y, parameters, lambd, layer_dims) # < = bug here

    #@timer
    def L_model_forward_multiclass(self, X, parameters):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
        
        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()
        
        Returns:
        AL -- last post-activation value
        caches -- list of caches containing:
                    every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
        """

        caches = []
        A = X
        L = len(parameters) // 2                  # number of layers in the neural network
        
        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            A_prev = A 
            ### START CODE HERE ### (≈ 2 lines of code)
            A, cache = self.linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
            
            caches.append(cache)
            ### END CODE HERE ###
        
        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        ### START CODE HERE ### (≈ 2 lines of code)
        AL, cache = self.linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "softmax")
        caches.append(cache)
        ### END CODE HERE ###
        
        #assert(AL.shape == (1,X.shape[1]))
                
        return AL, caches


    def L_model_backward(self, AL, Y, caches):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
        
        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                    the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
        
        Returns:
        grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ... 
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ... 
        """
        grads = {}
        L = len(caches) # the number of layers
        m = AL.shape[1]
        
        """convert Y to an array with zeroes of dimension(number of classes, m training examples) """
        Y = self.convert_y_to_binary_matrix(AL,Y)        
        
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
        
        # Initializing the backpropagation
        ### START CODE HERE ### (1 line of code)
        #dAL = -(np.divide(Y,AL) - np.divide(1-Y,1-AL))
        dZ = AL - Y
        ### END CODE HERE ###
        
        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
        ### START CODE HERE ### (approx. 2 lines)
        current_cache = caches[-1]
        #grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, activation = "softmax")
        grads["dA" + str(L-1)], grads["dW" +str(L)], grads["db" + str(L)] = self.linear_backward(dZ, current_cache[0])
        #print("last cache")
        ### END CODE HERE ###
        
        # Loop from l=L-2 to l=0
        for l in reversed(range(L-1)):
            # lth layer: (RELU -> LINEAR) gradients.
            # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
            ### START CODE HERE ### (approx. 5 lines)
            current_cache = caches[l]
            #print(f"cache {l}")
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l+1)], current_cache, activation = "relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
            ### END CODE HERE ###

        return grads

    def L_model_backward_with_regularisation(self, AL, Y, caches, lambd):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
        
        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                    the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
        
        Returns:
        grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ... 
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ... 
        """
        grads = {}
        L = len(caches) # the number of layers
        m = AL.shape[1]
        
        """convert Y to an array with zeroes of dimension(number of classes, m training examples) """
        Y = self.convert_y_to_binary_matrix(AL,Y)        
        
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
        
        # Initializing the backpropagation
        ### START CODE HERE ### (1 line of code)
        #dAL = -(np.divide(Y,AL) - np.divide(1-Y,1-AL))
        dZ = AL - Y
        ### END CODE HERE ###
        
        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
        ### START CODE HERE ### (approx. 2 lines)
        current_cache = caches[-1]
        #grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, activation = "softmax")
        grads["dA" + str(L-1)], grads["dW" +str(L)], grads["db" + str(L)] = self.linear_backward_with_regularisation(dZ, current_cache[0], lambd)
        #print("last cache")
        ### END CODE HERE ###
        
        # Loop from l=L-2 to l=0
        for l in reversed(range(L-1)):
            # lth layer: (RELU -> LINEAR) gradients.
            # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
            ### START CODE HERE ### (approx. 5 lines)
            current_cache = caches[l]
            #print(f"cache {l}")
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward_with_regularisation(grads["dA" + str(l+1)], current_cache, activation = "relu", lambd = lambd)
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
            ### END CODE HERE ###

        return grads
    
    def gradient_checking(self, epsilon = 1e-7): # doesn't work
        """
        performs gradient checking with l2 regularisation
        """
        def gradient_backprop(X, Y, parameters, lambd):
            
            ### perform manual gradient checking    ###
            AL, _ = softmax(np.dot(parameters["W1"], X) + parameters['b1'])
            m = AL.shape[1]            
            Y = self.convert_y_to_binary_matrix(AL,Y)
            test_cost = -1/m * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1-Y, np.log(1 - AL))) + lambd/(2*m) * np.square(np.linalg.norm(parameters["W1"]))
            dZ = AL - Y
            
            test_grads = {}
            test_grads["dW1"] = 1/m * np.dot(dZ, X.T) + lambd/m * parameters["W1"]
            test_grads["db1"] = 1/m * np.sum(dZ, keepdims = True, axis = 1)
            
            return test_cost, test_grads
            
        layers_dims = [2,1]
        np.random.seed(1)
        lambd = 0
        X = np.random.randn(2, 1)
        Y = np.random.choice([0, 1], size = (1, 1))
        print(f'X \n{X!r}')
        print(f'Y \n{Y!r}')
        parameters = self.initialize_parameters_deep(layers_dims)
        print(f'parameters \n{parameters!r}')
        
        test_cost, test_grads = gradient_backprop(X, Y, parameters, lambd)
        
        ###   determine gradient backprop  ###
        AL, caches = self.L_model_forward_multiclass(X, parameters)
        print(f'caches \n{caches!r}')
        print(f'AL \n{AL!r}')
        assert(AL.shape == self.convert_y_to_binary_matrix(AL,Y).shape)
        # Backward propagation.
        #grads = self.L_model_backward(AL, Y, caches)
        grads = self.L_model_backward_with_regularisation(AL, Y, caches, lambd)
        print(f'backprop grads\n {grads!r}')
        print(gradient_check(parameters, grads, X, Y, layers_dims, epsilon, hidden_layers_activation_fn="relu"))
        # compute cost
        #cost = self.compute_cost(AL, Y)
        cost = self.compute_cost_with_regularization(AL, Y, parameters, lambd, layers_dims)
        
        
        #
        ###   determine numerical estimation of grads   ###
        #epsilon = 1e-14
        numerical_grads = {}
        negParam = {}
        posParam = {}
        for param, vals in parameters.items():
            posParam[param] = np.copy(parameters[param])
            negParam[param] = np.copy(parameters[param])        
        
        for param, vals in parameters.items():
            rows, cols = vals.shape
            numerical_grads[f'd{param}'] = np.zeros((rows,cols))
            
            for r in range(0, rows):
                for c in range(0, cols):
                    posParam[param] = np.copy(parameters[param])                        
                    #print(f'param {param} r {r} c {c}')
                    #print(posParam)
                    #print(type(posParam))
                    #print(posParam['W1'])
                    print(epsilon)
                    posParam[param][r][c] +=  epsilon
                    print(parameters[param][r][c])
                    print(posParam[param][r][c])
                    print(parameters[param][r][c] == posParam[param][r][c])
                    #exit()
                    AL, caches = self.L_model_forward_multiclass(X, posParam)
                    #posCost = self.compute_cost(AL, Y)
                    
                    posCost = self.compute_cost_with_regularization(AL, Y, posParam, lambd, layers_dims)
                    posParam[param][r][c] -=  epsilon
                    
                    negParam[param] = np.copy(parameters[param])
                    
                    negParam[param][r][c] -= epsilon
                    print(parameters)
                    print(negParam)

                    print(f"negParam['W1'] == parameters['W1'] {negParam['W1'] == parameters['W1']}")
                    print(f"negParam['b1'] == parameters['b1'] {negParam['b1'] == parameters['b1']}")
                    print(parameters[param][r][c])
                    print(negParam[param][r][c])
                    print(parameters[param][r][c] == negParam[param][r][c])
                    #exit()
                    AL, caches = self.L_model_forward_multiclass(X, negParam)
                    
                    #negCost = self.compute_cost(AL, Y)
                    
                    negCost = self.compute_cost_with_regularization(AL, Y, negParam, lambd, layers_dims)
                    negParam[param][r][c] += epsilon
                    
                    assert(negParam[param][r][c] == parameters[param][r][c])
                    assert(posParam[param][r][c] == parameters[param][r][c])
                    numgrad = (posCost - negCost) / (2 * epsilon)
                    
                    numerical_grads[f'd{param}'][r][c] = numgrad
                    
        
        print(f'backprop grads\n {grads!r}')            
        print(f'numerical grads \n{numerical_grads!r}')
        print(posCost)
        print(negCost)
        print(cost)
        print(test_cost)
        
        grads = gradients_to_vector(grads)
        numerical_grads = gradients_to_vector(numerical_grads)
        test_grads = gradients_to_vector(test_grads)
        
        print(f'backprop grads\n {grads!r}')            
        print(f'numerical grads \n{numerical_grads!r}')
        print(f'test grads \n{test_grads!r}')
        error = np.linalg.norm(grads - numerical_grads)/ (np.linalg.norm(grads) + np.linalg.norm(numerical_grads))
        print(f'error {error}')
        return error
    def L_layer_model_multiclass(self, X, Y, layers_dims, learning_rate = 0.009, lambd = 0, num_iterations = 3000, print_cost=False, plot_cost = False, load_weights = False, save_weights = False, save_hyperparameters = False, keep_results = False):
        """
        Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SOFTMAX.
        
        Arguments:
        X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector, of shape (1, number of examples)
        layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
        learning_rate -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop
        print_cost -- if True, it prints the cost every 100 steps
        
        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """

        np.random.seed(1)
        costs = []                         # keep track of cost
        self.layers_dims = layers_dims
        self.keep_results = keep_results
        # Parameters initialization. 
        if load_weights:
            parameters = self.load_weights(layers_dims)
            print("loading weights")
        else:
            parameters = self.initialize_parameters_deep(layers_dims)
            print("initiating random initialisation")
                
        # Loop (gradient descent)
        for i in range(0, num_iterations):

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SOFTMAX.
            ### START CODE HERE ### (≈ 1 line of code)
            AL, caches = self.L_model_forward_multiclass(X, parameters)
            ### END CODE HERE ###
            
            # Compute cost.
            ### START CODE HERE ### (≈ 1 line of code)
            #cost = self.compute_cost(AL, Y)
            cost = self.compute_cost_with_regularization(AL, Y, parameters, lambd, layers_dims)
            #print(f"cost1 - cost {cost1-cost}")
            ### END CODE HERE ###
        
            # Backward propagation.
            ### START CODE HERE ### (≈ 1 line of code)
            #grads = self.L_model_backward(AL, Y, caches)
            grads = self.L_model_backward_with_regularisation(AL, Y, caches, lambd)
            #print("grads1 - grads {}".format(grads1["dW2"] - grads["dW2"]))
            ### END CODE HERE ###
     
            # Update parameters.
            ### START CODE HERE ### (≈ 1 line of code)
            parameters = self.update_parameters(parameters, grads, learning_rate)
            ### END CODE HERE ###
            
            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))#, flush = True)
                
            if print_cost and i % 100 == 0:
                costs.append(cost)
            
            """
            if print_cost:
                self.disp_cost(i, cost)  """  
        
        if costs[0] > costs[-1]: # checking if the model improved
            #saves the weights.
            if save_weights:
                self.save_weights(layers_dims, parameters)
            if save_hyperparameters:
                hyperparameters = {"learning_rate": learning_rate, "num_iterations (hundreds)" : num_iterations/100, "lambda" : lambd, "cost" : costs[-1]} #costs[-1]
                if load_weights:
                    self.save_hyperparameters(layers_dims, hyperparameters, reset = False)
                else:
                    self.save_hyperparameters(layers_dims, hyperparameters, reset = True)
            if self.keep_results:
                self.results["costs"] = costs
                if load_weights:
                    self.reset_results = False
                else:
                    self.reset_results = True
        else:
            print("the model did not improve, hence the weights were not changed")
            self.keep_results = False
        
        
        
        # plot the cost        
        if plot_cost:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per hundreds)')
            plt.title(f"Learning rate = {learning_rate}, lambda = {lambd}")
            plt.show()
        
        return parameters  
    
    def predict(self, X, y, parameters, name = ""):
    
        """
        This function is used to predict the results of a  L-layer neural network.
        
        Arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model
        
        Returns:
        probas -- predictions fpr the given dataset X (1,m)
        """
        
        m = X.shape[1]
        n = len(parameters) // 2 # number of layers in the neural network
        p = np.zeros((1,m))
        
        # Forward propagation
        probas, caches = self.L_model_forward_multiclass(X, parameters)
        
        #print(probas.shape) # (number of classes, number of examples in set
        #print(probas)
        
        probas = np.argmax(probas, axis = 0).reshape(1,y.shape[1])
        
        print(f"probas: {probas}")
        
        print(f"y: {y}")
        """
        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
        """
        #print results
        #print ("predictions: " + str(p))
        #print ("true labels: " + str(y))
        print("Number correct: " + str(np.sum((probas==y))))
        accuracy = str(np.sum((probas == y)/m))
        print("Accuracy: "  + str(np.sum((probas == y)/m)))
        
        self.results[f"accuracy_{name}"] = accuracy
            
        return probas
    
    def confusion_matrix(self, true_labels, predicted, classes, name = ""):
        """
        This function is used to display the confusion matrix
        
        Arguments:
        true_labels -- labels of y, of shape (1,m)
        preducted -- predicted values y-hat, of shape (1,m)
        classes -- a list of classes decoded or not decoded with shape (1, num_classes)
        
        Returns:
        confusion -- a (num_classes, num_classes) matrix
        for a classes.size = 6:
        the confusion matrix looks liek this:
                     true_label 
                    0,0,0,0,1,2
          predicted 1,4,2,1,4,1
           calues   4,5,1,3,5,1
                    3,5,1,6,3,6
                    0,0,2,5,1,6
        """    
        confusion = np.zeros((classes.size, classes.size), dtype = np.int16) # initialise with zeros, int16 is specified to ensure the confusion matrix is not a float
        predicted = predicted.astype(int) # need to convert to int to make sure the index are integers rather than floats (literally figured it out in 5 minutes, hah!)
        true_labels = true_labels.astype(int)
        #print(confusion)
        element_list = []
        """
        print(f"type(predicted) {type(predicted)}")
        print(f"type(true_labels) {type(true_labels)}")
        print(f"predicted.shape {predicted.shape}")
        print(f"true_labels.shape {true_labels.shape}")
        """
        """
        for elem in zip(np.squeeze(predicted), np.squeeze(true_labels)): # zip combines elements from each array into tuples, np.squeeze removes that unesseccary dimension (took 30 minutes to debug) 
            element_list.append(elem)
        print(element_list)"""
        for elem in zip(np.squeeze(predicted), np.squeeze(true_labels)): # zip combines elements from each array into tuples , np.squeeze removes that unesseccary dimension (m,1) => (m,) and allows tuples to be formed(took 30 minutes to debug)
            confusion[elem] +=1
        
        self.results[f"confusion_{name}"] = confusion
        return(confusion)
        
if __name__ == '__main__':
    if program == "insect":
        with h5py.File(dataset_path, 'r') as hdf:
            ls = list(hdf.keys()) 
            print(f"List of datasets in this file: \n {ls}")
            
            x_orig = np.array(hdf.get('set_x'))
            y_orig = np.array(hdf.get('set_y')).reshape((1,-1)) # changes (8,) to(1,8)
            classes = np.array(hdf.get('list_classes'))
            
        print(f"x_orig.shape: {x_orig.shape}")
        print(f"y_orig.shape: {y_orig.shape}")
        print(f"classes: {[x.decode('utf-8') for x in classes]}")
        print(f"number of insects per class {list(zip(*np.unique(y_orig, return_counts = True)))}") #np.unique returns a tuple, thus * is needed to tell zip to is a tuple with two args (unique, counts), list consumes the iterable produced by zip
        print(f"y_orig: {y_orig}");


        
        insect_dnn = MulticlassDnn()
        
        ratios = [60,20,20]
        
        train_x_orig, train_y, val_x_orig, val_y, test_x_orig, test_y = insect_dnn.train_val_test(ratios, x_orig, y_orig)
        
        m_train = train_x_orig.shape[0]
        num_px = train_x_orig.shape[1]
        m_test = test_x_orig.shape[0]
        m_val = val_x_orig.shape[0]
        print("Checking the dataset")
        print ("Number of training examples: " + str(m_train))
        print ("Number of validation examples: " + str(m_val))
        print ("Number of testing examples: " + str(m_test))
        print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
        print ("train_x_orig shape: " + str(train_x_orig.shape))
        print ("train_y shape: " + str(train_y.shape))
        print ("val_x_orig shape: " + str(val_x_orig.shape))
        print ("val_y shape: " + str(val_y.shape))    
        print ("test_x_orig shape: " + str(test_x_orig.shape))
        print ("test_y shape: " + str(test_y.shape))
        print()
        
        

        #insect_dnn.test_match(classes, test_x_orig, test_y)
        
        
        # Reshape the training, validation and test examples 
        train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
        val_x_flatten = val_x_orig.reshape(val_x_orig.shape[0],-1).T
        test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
        
        # Standardize data to have feature values between 0 and 1.
        train_x = train_x_flatten/255.
        val_x = val_x_flatten/255.
        test_x = test_x_flatten/255.
        
        print("checking the flattened shape")
        print ("train_x's shape: " + str(train_x.shape))
        print ("val_x's shape: " + str(val_x.shape))
        print ("test_x's shape: " + str(test_x.shape))
        

        # define constants for the model:

        layers_dims = [train_x.shape[0], 20, 7, 5, len(classes)] # 4 layer model
        print(f"layer dimensions: {layers_dims}")
        
        
        #train the model
        
        parameters = insect_dnn.L_layer_model_multiclass(train_x, train_y, layers_dims, learning_rate = 0.0008, lambd = 2, num_iterations = 800,
                                                            print_cost = True, plot_cost = True, load_weights = False, save_weights = True,
                                                            save_hyperparameters = True, keep_results = True) #0.0008 is the original
     
        #parameters = insect_dnn.load_weights(layers_dims) #toggle between this line and the one above
        
        print("train set")
        pred_train = insect_dnn.predict(train_x, train_y, parameters, name = "train")
        print(insect_dnn.confusion_matrix(train_y, pred_train, classes, name = "train"))
        
        print("validation set")
        pred_val = insect_dnn.predict(val_x, val_y, parameters, name = "val")
        print(insect_dnn.confusion_matrix(val_y, pred_val, classes, name = "val"))
        
        error_directory = "error_analysis"
        
        #insect_dnn.error_analysis(pred_val, val_x, val_y, classes, num_px, error_directory) #in a directoy, uploads files that were labelled incorrectly
        insect_dnn.call_destructor() # don't use __del__, it doesn't let me write to the files for some reason

    elif program == "identify_insect":
       
        result_dict = {"rank":{}, "class":{}, "probabilities":{}}
        insect_dnn = MulticlassDnn()
        with h5py.File(dataset_path, 'r') as hdf:
            classes = np.array(hdf.get('list_classes'))
            classes = [x.decode('utf-8') for x in classes]
        
        layers_dims = [67500, 20, 7, 5, len(classes)]
        
        num_px = int((layers_dims[0]/3)**0.5)
        
        parameters = insect_dnn.load_weights(layers_dims)
        
        print(classes) # encoded in utf-8

        photo_where = "file" # haven't used
        file_name = "aphid_train.jpg"
        
        image = np.array(plt.imread(file_name)) # height x width x depth
        resized_image = np.array(Image.fromarray(image).resize((num_px,num_px)))
        
        """ #testing if images have been resized correctly
        fig = plt.figure(figsize = (8,8))
        fig.add_subplot(1,2,1)
        plt.imshow(image)
        fig.add_subplot(1,2,2)
        plt.imshow(resized_image)
        plt.show()
        """
        image_flatten = resized_image.reshape(1, -1).T
        
        image_standardised = image_flatten/255.

        print("checking the flattened shape")
        print ("image_standardised's shape: " + str(image_standardised.shape))
        
        print(f"layer dimensions: {layers_dims}")
        
        probabilities, _ = insect_dnn.L_model_forward_multiclass(image_standardised, parameters)
        
        sort_probabilities_reverse = np.argsort(np.argsort(probabilities.reshape(probabilities.shape[0])))

        print(f"sort_probabilities_reverse {sort_probabilities_reverse}")

        sort_probabilities = len(sort_probabilities_reverse) - sort_probabilities_reverse
        

        print(f"sort_probabilities.shape {sort_probabilities.shape}")
        print(f"type(sort_probabilities) {type(sort_probabilities)}")
        
        sort_probabilities = sort_probabilities.tolist()
        print(sort_probabilities)
        
        probabilities = probabilities.reshape(probabilities.shape[0]).tolist()
        print(type(probabilities))
        print(probabilities)
        
        result_dict["probabilities"] = dict(enumerate(probabilities))
        result_dict["class"] = dict(enumerate(classes))
        result_dict["rank"] = dict(enumerate(sort_probabilities))
        print(result_dict)
        
        
    elif program == "cat":
        train_dataset = h5py.File('train_catvnoncat.h5', "r")
        train_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
        train_y = np.array(train_dataset["train_set_y"][:]) # your train set labels

        test_dataset = h5py.File('test_catvnoncat.h5', "r")
        test_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
        test_y = np.array(test_dataset["test_set_y"][:]) # your test set labels

        classes = np.array(test_dataset["list_classes"][:]) # the list of classes
        
        train_y = train_y.reshape((1, train_y.shape[0]))
        test_y= test_y.reshape((1, test_y.shape[0]))
        
        #print example of a picture:
        """
        index = 10
        plt.imshow(train_x_orig[index])
        plt.show()
        print("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")
        """
        # Checking the dataset 
        m_train = train_x_orig.shape[0]
        num_px = train_x_orig.shape[1]
        m_test = test_x_orig.shape[0]
        print("Checking the dataset")
        print ("Number of training examples: " + str(m_train))
        print ("Number of testing examples: " + str(m_test))
        print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
        print ("train_x_orig shape: " + str(train_x_orig.shape))
        print ("train_y shape: " + str(train_y.shape))
        print ("test_x_orig shape: " + str(test_x_orig.shape))
        print ("test_y shape: " + str(test_y.shape))
        print()
        
        
        #define constants for the model:
        
        layers_dims = [12288, 20, 7, 5, 1] # 4 layer model
        
        
        # Reshape the training and test examples 
        train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
        test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

        # Standardize data to have feature values between 0 and 1.
        train_x = train_x_flatten/255.
        test_x = test_x_flatten/255.
        print("checking the flattened shape")
        print ("train_x's shape: " + str(train_x.shape))
        print ("test_x's shape: " + str(test_x.shape))
        
        print("Trained parameters")
        
        cat_dnn = Dnn()
        parameters = cat_dnn.L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)
        
        print("train set")
        pred_train = cat_dnn.predict(train_x, train_y, parameters)
        
        print("test set")
        pred_test = cat_dnn.predict(test_x, test_y, parameters)

        
        
    elif program == "confusion_testing":
        confusion = MulticlassDnn();
        #test case 1:
        true_labels = np.array([[0., 1., 2., 0., 0., 1., 1.]]).T
        predicted = np.array([[2., 0., 1., 0., 1., 2., 0.]]).T
        list_classes = ["aphid", "helicoverpa", "beaglebone"]
        classes = np.array([x.encode('utf-8') for x in list_classes])
        print(f"test case 1: \n{confusion.confusion_matrix(true_labels, predicted, classes)}")
        """
        [[1. 2. 0.]
         [1. 0. 1.]
         [1. 1. 0.]]
        """
        #test case 2:
        true_labels = np.array([[0.,2.,3.,2.,1.,2.,4.,4.,2.,3.,1.,4.]]).T
        predicted = np.array([[1.,1.,1.,1.,1.,2.,2.,2.,2.,3.,4.,2.]]).T
        list_classes = ["aphid", "helicoverpa", "fall army worm", "australian plague locust", "fig leaf beetle"]
        classes = np.array([x.encode('utf-8') for x in list_classes])
        print(f"test case 2: \n{confusion.confusion_matrix(true_labels, predicted, classes)}")   
        """
        [[0. 0. 0. 0. 0.]
         [1. 1. 2. 1. 0.]
         [0. 0. 2. 0. 3.]
         [0. 0. 0. 1. 0.]
         [0. 1. 0. 0. 0.]]
        """
        
    elif program == "save_weights_testing":
        weight_savin = MulticlassDnn()
        layer_dims = [100,20,7,3]
        weights = {"W1": np.arange(2000).reshape(20,100), "b1" : np.arange(20).reshape(20,1), "W2":np.random.randn(7,20), "b2" : np.zeros((7,1))}
        weight_savin.save_weights(layer_dims, weights)
        
    elif program == "load_weights_testing":
        # prerequisites: run program with program = "save_weights_testing"
        weight_loadin = MulticlassDnn()
        #layer_dims = [100,20,7,3]
        layer_dims = [67500, 20, 7, 5, 4]
        print(weights)
        
    elif program == "save_hyperparameters_testing":
        hyperparameter_savin = MulticlassDnn()
        layer_dims = [100, 20,7,3]
        hyperparameters = {"learning_rate": 0.0008, "num_iterations (hundreds)" : 1000}
        hyperparameter_savin.save_hyperparameters(layer_dims, hyperparameters, reset = True)
        
    elif program == "load_hyperparameters_testing": # can be used to load the model records if u want
        hyperparameter_loadin = MulticlassDnn()
        #layer_dims = [100, 20,7,3]
        layer_dims = [67500, 20, 7, 5, 4] # actual dimensions don't use unless you know what you are doing
        a = hyperparameter_loadin.load_hyperparameters(layer_dims)
        print(a)
        
    elif program == "gradient_checking":
        gradient_checkin = MulticlassDnn()
        """
        epsilons = [10**(-l) for l in range(7, 15)]
        from joblib import Parallel, delayed
        errors = Parallel(n_jobs = 8)(delayed(gradient_checkin.gradient_checking)(epsilon) for epsilon in epsilons)
        plt.plot(list(map(np.log10,epsilons)), errors)
        plt.xlabel('log10 epsilons')
        plt.ylabel('error')
        plt.show()
        """
        gradient_checkin.gradient_checking(epsilon = 1e-7)
    elif program == "decision_boundary_2D":
        
        csv_file = 'labelled_data2D_3.csv'
        import pandas as pd
        
        def feature_norm(X):
           
            m = X.shape[1]
            
            mu = np.mean(X, axis = 1, keepdims = True)
            
            
            sigma = np.std(X, axis = 1, keepdims = True)
                        
            X_norm = np.divide((X - mu), sigma)
            
            return X_norm, mu, sigma
        
        plots = pd.read_csv(csv_file)
        plots = plots.to_numpy()
        
        X = plots[:,[0,1]].T
        
        X, mu, sigma = feature_norm(X)
        
        Y = plots[:,-1].astype(np.uint8).reshape(1,-1)
        
        
        
        
        LABEL_COLOR_MAP = {0 : 'r',
                           1 : 'k'
                           }
        labels = Y.tolist()
        label_color = [LABEL_COLOR_MAP[l] for l in labels[0]]
        
        
        
        plt.figure(2)
        
        
        decision_bound = Dnn()
        layers_dims = [X.shape[0], 20, 7, 5, 1]

        parameters = decision_bound.L_layer_model(X,
                                       Y, 
                                       layers_dims = layers_dims,
                                       learning_rate = 0.001, 
                                       num_iterations = 3000000,
                                       print_cost=True, 
                                       lambd = 0.0,
                                       show_plot = True) #149300
        """                               
        parameters = decision_bound.L_layer_model(X,
                                       Y, 
                                       layers_dims = layers_dims,
                                       learning_rate = 0.0001, 
                                       num_iterations = 85200,
                                       print_cost=True, 
                                       lambd = 0,
                                       parameters = parameters,
                                       show_plot = False)
        parameters = decision_bound.L_layer_model(X,
                                       Y, 
                                       layers_dims = layers_dims,
                                       learning_rate = 0.0000001, 
                                       num_iterations = 70000,
                                       print_cost=True, 
                                       lambd = 0,
                                       parameters = parameters,
                                       show_plot = True)        
        
        parameters = decision_bound.L_layer_model(X,
                                       Y, 
                                       layers_dims = layers_dims,
                                       learning_rate = 0.001, 
                                       num_iterations = 10000,
                                       print_cost=False, 
                                       lambd = 0,
                                       parameters = parameters)
        """
        print(parameters)
        probas = decision_bound.L_model_forward(X, parameters)[0]
        
        labels = (probas >= 0.5) * 1
        accuracy = np.mean(labels == Y) * 100
        print(accuracy)
        decision_bound.predict(X, Y, parameters)
        
        def plot_decision_boundary(X, Y, layers_dims, parameters):
            plt.figure(1)
            
            X1 = X[0,:]
            X2 = X[1,:]
            assert X.shape[0] == 2
            
            #print(parameters)
            
            AL, caches = decision_bound.L_model_forward(X, parameters)
            probas = sigmoid(AL)[0]
            
            
            num_val = 100
            
            x1 = np.linspace(np.min(X1), np.max(X1), num = num_val)
            x2 = np.linspace(np.min(X2), np.max(X2), num = num_val)
            
            xx, yy = np.meshgrid(x1, x2);
            
            X = np.concatenate((xx.reshape(1, -1), yy.reshape(1,-1)), axis = 0)
            
            AL, caches = decision_bound.L_model_forward(X, parameters)
            
            P = AL
            
            print(P)
            
            cs = plt.contourf(xx, yy, P.reshape(xx.shape), cmap=plt.cm.Paired, levels = [0, 0.5, 1])
            plt.colorbar(cs)
            
            plt.figure(3)
            cs2 = plt.contourf(xx, yy, P.reshape(xx.shape), cmap=plt.cm.Paired, levels = [0, 0.5, 1])
            plt.colorbar(cs2)
            
        plot_decision_boundary(X, Y, layers_dims, parameters)
        
        
        
        plt.figure(1)
        """
        x = np.linspace(np.min(X[0, :]), np.max(X[0, :]), num = 100)
        y = - parameters['W1'][0,0] / parameters['W1'][0,1] * x - parameters['b1'] / parameters['W1'][0,1]
        
        plt.plot(x,y.reshape(x.shape), '-')
        """
        plt.scatter(X[0,:], X[1,:], c = label_color)
        plt.show()
        """
        with open(file_name, 'r') as csvfile:
            plots = csv.reader(csvfile, delimiter = ',')
            print(plots)
            print(type(plots))
            plots = np.array(plots)
            print(plots.shape)
        """