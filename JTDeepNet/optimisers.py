
#================================================================
#
#   File name   : optimisers.py
#   Author      : Josiah Tan
#   Created date: 24/06/2020
#   Description : contains all the optimisers u need man
#
#================================================================

#================================================================
import numpy as np

class Optimiser:
    pass

class GradientDesc(Optimiser):
    def __init__(self, learning_rate = 0.001):
        self.learning_rate = learning_rate
    def update(self, obj):
        """
        updates param based on their partial derivatives
        note: params and grads elements must be arranged in the same order
        parameterss -- params: parameters given in tuple form, e.g (W, b)
                    -- grads: partial derivatives given in tuple form, e.g (dW, db)
        returns     -- new_params
        """
        #parameters, gradients = obj.parameters, obj.gradients
        for parameter, gradient in zip(obj.parameters, obj.gradients):
            parameter -= self.learning_rate * gradient

            
        #return parameters
        
class Adam(Optimiser):
    def __init__(self, learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon=1e-07):
        #hyperparameters
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.iteration_count = 1 # used for correction value with momentum and rms

        
    def initialise(self, obj):
        #initialsie constants for RMS and Momentum components
        obj.__Vs = [] # momentum, this variable can be accessed by (sweet name mangling): obj._Adam__Vs
        obj.__Ss = [] # RMS, obj._Adam__Ss
        
        for parameter in obj.parameters:
            obj.__Vs.append(np.zeros(np.shape(parameter))) # for Linear object, __Vs = (Vdw, Vdb)
            obj.__Ss.append(np.zeros(np.shape(parameter)))
        
    def update(self, obj):
        if not obj.optimiser_initialised: # only want to run initialise once
            self.initialise(obj)
            obj_initialised = True
            
        for V, S, parameter, gradient in zip(obj.__Vs, obj.__Ss, obj.parameters, obj.gradients):
            #print(f'V before {V}')
            V = self.beta_1 * V + (1 - self.beta_1) * gradient
            #print(f'V after {V}')
            S = self.beta_2 * S + (1 - self.beta_2) * gradient ** 2
            
            V_corrected = V / (1 - self.beta_1 ** self.iteration_count)
            
            S_corrected = S / (1 - self.beta_2 ** self.iteration_count)
            
            parameter -= self.learning_rate * V_corrected / (np.sqrt(S_corrected) + self.epsilon)
            
            self.iteration_count += 1
        
        