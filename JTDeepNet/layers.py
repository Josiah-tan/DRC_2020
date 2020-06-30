
#================================================================
#
#   File name   : layers.py
#   Author      : Josiah Tan
#   Created date: 23/06/2020
#   Description : Used to create deep net layers, like batch norm, linear, conv, max pool, WHO knows???
#
#================================================================

#================================================================
from .name import NameManager
import numpy as np

class Layer(NameManager):
    def __init__(self):
        super().__init__()
        self.trainable = True # use True to denote the need to train the weights
        
class Linear(Layer):
    name = "linear"
    
    def __init__(self, output_dims = (10, None), initialiser = "glorot", name = ""):
        super().__init__()
        self.output_dims = output_dims
        
        self.output_size = output_dims[0]
        
        self.initialiser = initialiser
        
        Linear.change_name(name)

    def initialise(self, prev):
        """ initialise parameters of the linear layer """
        if self.initialiser == "glorot":
            output_size = self.output_size
            input_size = prev.output_size
            
            self.W = np.random.randn(output_size, input_size) * 2 / input_size 
            self.b = np.zeros((output_size, 1))
            
            self.dW = np.zeros(self.W.shape) # initialising derivatives (important since we need their shapes)
            self.db = np.zeros(self.b.shape)
                        
            self.regularisable_params = (self.dW) # for regularisation computation later
            
            self.parameters = (self.W, self.b) # initialise parameters for updating
            self.gradients = (self.dW, self.db) # initialise partial derivatives for updating
            
            self.optimiser_initialised = False # need to initialise V and S for momentum and rms prop
    def forward(self, prev):
        self.A = prev.fwd_output
        self.Z = np.dot(self.W, self.A) + self.b
        self.fwd_output = self.Z
    
    def backward(self, prev):
        m = self.jtdnn_obj.m
        lambd = self.jtdnn_obj.lambd
        self.dZ = prev.bwd_output
        
        self.dW = 1/m * np.dot(self.dZ, self.A.T) + lambd / m * self.W
        self.db = 1/m * np.sum(self.dZ, axis = 1, keepdims = True)
        self.dA = np.dot(self.W.T, self.dZ)
        
        
        assert (self.dA.shape == self.A.shape)
        assert (self.dW.shape == self.W.shape)
        assert (self.db.shape == self.b.shape)
        
        self.bwd_output = self.dA
        
    def update(self):
        
        self.parameters = (self.W, self.b) # need to somehow make elements here that are references not copies, but idk how
        try:
            self.gradients = (self.dW, self.db)
        except:
            print(self.key_name)
        
        self.jtdnn_obj.optimiser.update(self)
    
    def __call__(self, prev):
        
        self.initialise(prev)
          
        #print(prev.jtdnn_obj) # JTDNN object
        self.jtdnn_obj = prev.jtdnn_obj # make a reference to the original jtdnn_obj
        
        Linear.add_name(self) # add this instance to the dictionary

        return self
        

