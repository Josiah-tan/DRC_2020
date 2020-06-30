
#================================================================
#
#   File name   : losses.py
#   Author      : Josiah Tan
#   Created date: 23/06/2020
#   Description : Used to create deep net loss functions such as categoricalCrossEntropy, binaryCrossEntropy
#
#================================================================

#================================================================
import numpy as np
import matplotlib.pyplot as plt

from .layers import Layer
#import activations
    
class Loss:
    def __init__(self, jtdnn_obj, store_cost = True, fig_num = 1):
        self.jtdnn_obj = jtdnn_obj
        self.store_cost = store_cost
        self.fig_num = fig_num
        
        if store_cost:
            self.costs = []
            
    def add_cost(self, cost):
        if self.store_cost:
            self.costs.append(cost)
        else:
            raise ("You need to seet store_cost to True to store cost values")
    
    def plot_cost(self, title = "Cost per Iteration", xlabel = "Number of number of iterations (100s)", ylabel = "Cost"):
        plt.figure(self.fig_num)
        plt.plot(self.costs)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)


class BinaryCrossEntropy(Loss):
    name = "BinaryCrossEntropy"
    def __init__(self, jtdnn_obj, store_cost = True, fig_num = 1):
        super().__init__(jtdnn_obj, store_cost, fig_num)
     
    def compute_cost(self, Y, predictions, lambd):
        """have not yet implemented regularisation"""
        
        m = Y.shape[1]
        
        sum_regularisable_param_square = 0
        
        for obj_str in self.jtdnn_obj.graph_lis[self.jtdnn_obj.start + 1 : self.jtdnn_obj.end + 1]: # loop should be same as fwd prop
            #print(f'forward {obj_str}')
            obj = self.jtdnn_obj.graph_dict[obj_str]
            if isinstance(obj, Layer): # assuming only Layer objects contain regularisable params
                for regularisable_param in obj.regularisable_params:
                    sum_regularisable_param_square += np.sum(np.square(regularisable_param))
        
        losses = np.multiply(Y,np.log(predictions))+np.multiply(1-Y,np.log(1-predictions))
        
        """This checks if any elements are NAN, if they are, it also checks if y^i_j == al^i_j since 0log0 = 0 not nan"""
        if np.isnan(losses).any():
            #print(f'losses {losses}')
            #print(f'np.isnan(losses) \n{np.isnan(losses)}')
            #print(f'Y == AL \n{Y == AL}')
            #print(f'(Y == AL) * np.isnan(losses) \n{(Y == AL) * np.isnan(losses)}')
        
            losses[(Y == predictions) * np.isnan(losses)] = 0
        
            #print(f'new losses {losses}')
        
        cross_entropy_cost = np.squeeze(-1/m * np.sum(losses))
        
        L2_regularization_cost = lambd * (sum_regularisable_param_square) / (2 * m)
        
        cost = cross_entropy_cost + L2_regularization_cost
        
        self.add_cost(cost)
        
        return cost
        
    
    






