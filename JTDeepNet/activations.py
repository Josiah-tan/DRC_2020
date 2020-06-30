
#================================================================
#
#   File name   : layers.py
#   Author      : Josiah Tan
#   Created date: 23/06/2020
#   Description : Used to create deep net activation functions such as sigmoid, tanh, relu, leakyRelu, softmax
#
#================================================================

#================================================================
import numpy as np

from .name import NameManager

class Activation(NameManager):
    def __init__(self):
        super().__init__()
        self.trainable = False # I'm pretty sure u don't train the activation layers lol, but I put it here just in case
    
class Sigmoid(Activation):
    name = 'sigmoid'
    def __init__(self, prev, name = 'sigmoid'):
        super().__init__()
        self.jtdnn_obj = prev.jtdnn_obj # make a reference to the original jtdnn_obj
        
        self.input_size = prev.output_size
        
        self.output_size = self.input_size
        
        Sigmoid.change_name(name)
        
        Sigmoid.add_name(self)
        
    
    def forward(self, prev):
        self.Z = prev.fwd_output
        self.A = 1 / (1 + np.exp(-self.Z))
        self.fwd_output = self.A
        
    def backward(self, prev):
        self.dA = prev.bwd_output
        self.dZ = self.A * (1 - self.A) * self.dA
        self.bwd_output = self.dZ
        
        
class ReLu(Activation):
    #man having sigmoid as a template makes this implementation ezzz
    name = "relu"
    def __init__(self, prev, name = 'relu'):
        super().__init__()
        self.jtdnn_obj = prev.jtdnn_obj # make a reference to the original jtdnn_obj
        
        self.input_size = prev.output_size
        
        self.output_size = self.input_size
        
        ReLu.change_name(name)
        
        ReLu.add_name(self)
        
    
    def forward(self, prev):
        self.Z = prev.fwd_output
        self.A = np.maximum(0, self.Z)
        self.fwd_output = self.A
        
    def backward(self, prev):
        self.dA = prev.bwd_output
        self.dZ = np.array(self.dA, copy=True) # just converting dz to a correct object.
        # When z <= 0, you should set dz to 0 as well. 
        self.dZ[self.Z <= 0] = 0
        self.bwd_output = self.dZ



