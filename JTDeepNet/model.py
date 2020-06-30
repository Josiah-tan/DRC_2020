
#================================================================
#
#   File name   : sequential.py
#   Author      : Josiah Tan
#   Created date: 23/06/2020
#   Description : Used to create DNN models maybe CNN in the future?
#
#================================================================

#================================================================

#importing third partay packages
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import joblib
import pandas as pd

if __name__ == "__main__":
    #importing my dnn modules
    import layers
    import activations
    import optimisers
    import losses

    from random_utils import feature_norm, plot_decision_boundary, mini_batch_generator


class JTDNN:
    """Used to build deep neural networks"""
    def __init__(self):
        self.graph_lis = [] # used to store lists of keys e.g linear1, linear2
        self.graph_dict = OrderedDict() # used to store keys and their respective objects/ instances in order of fwd prop
    
    def compile(self, input = None, output = None, lambd = 0.01, loss = "BinaryCrossEntropy", metrics = "accuracy", optimiser = "BGD"): #loss = "BinaryCrossEntropy", metrics = "accuracy", optimiser = "BGD" hasn't really been implemented yet
        """This compiles, hence sets the paramaeters for the model"""
        #initialise constants
        if input is not None:
            self.input = input
        if output is not None:
            self.output = output
        self.lambd = lambd
        
        
        #initialise objects
        self.optimiser = optimiser
        self.loss = loss
        
        self.initialise_start_2_end()
    def compute_cost(self, Y = None):
        """
        computes the cost of the function based on what the user compiled as 'loss'
        
        parameters: Y -- binary matrix, shape (n, m)
        
        returns: self.cost -- result of the cost functions
        
        other dependencies: self.predictions -- from forward_prop
            
        """
        if Y is None:
            try:
                Y = self.Y
            except:
                raise("Missing input argument Y")
        else:
            self.Y = Y
        self.current_cost = self.loss.compute_cost(Y, self.predictions, self.lambd)
        
        return self.current_cost
    
    def get_costs(self):
        return self.loss.costs
    
    def plot_cost(self, title = "Cost per Iteration", xlabel = "Number of number of iterations (100s)", ylabel = "Cost"):
        self.loss.plot_cost(title, xlabel, ylabel)
    
    def forward_prop(self, X):
        """calculates all predeterminate values and returns the final predictions"""
        prev_obj = self.input_obj
        
        prev_obj.fwd_output = X
        for obj_str in self.graph_lis[self.start + 1 : self.end + 1]:
            #print(f'forward {obj_str}')
            obj = self.graph_dict[obj_str]
            obj.forward(prev_obj)
            
            prev_obj = obj
            
            
        
        self.predictions = obj.fwd_output
        return self.predictions
        
    def back_prop(self, Y = None):
        """calculates all the partial derivatives"""
        if Y is None:
            try:
                Y = self.Y
            except:
                error("Missing input argument Y")
        else:
            self.Y = Y
        self.m = Y.shape[1]
        if self.loss.name == "BinaryCrossEntropy":
            #assert isinstance(self.graph_dict[self.graph_lis[-1]], activations.Sigmoid) # check if last activation is sigmoid
            #assert isinstance(self.output, activations.Sigmoid) # check if the output function is a sigmoid
            
            prev_obj = self.output # changed from self.graph_dict[self.graph_lis[-1]] since we want a specific output, not just the final layer # note that prev_obj has a completely different meaning for back_prop alright?
            prev_obj.bwd_output = self.predictions - Y # dZ = AL - Y
            
            graph_lis_len = len(self.graph_lis)
            
            for obj_str in self.graph_lis[self.end - 1 - graph_lis_len:self.start - graph_lis_len:-1]: # note that -ve index is being used cause u can't do [5:-1:-1] to get the first element. Also it generally shouldn't perform backprop on input object so it is excluded
                #print(f'backward {obj_str}')
                obj = self.graph_dict[obj_str]
                           
                obj.backward(prev_obj)
                
                prev_obj = obj
               
    def update_weights(self):
        """updates all the weights of the NN"""
        for obj_str in self.graph_lis[self.start + 1: self.end + 1]:
            #print(f'update {obj_str}')
            obj = self.graph_dict[obj_str]
            
            if obj.trainable:
                obj.update()
               
            
    def input(self, input_dims):
        """Creates an input object based on the input dimensions and returns this input object"""
        self.input_obj = Input(input_dims, self)
         
        return self.input_obj
        
    
    def initialise_start_2_end(self):
        """finds the indexes of both the starting object and the last object corresponding to self.input and self.output respectively"""
        self.end = list(self.graph_dict.values()).index(self.output) # finds index of output instance
        
        if isinstance(self.input, Input): 
            self.start = -1 # not a typo
        else:
            self.start = list(self.graph_dict.values()).index(self.input)
class Input:
    def __init__(self, input_dims, jtdnn_obj):
        #assume that input_dims = (n, None), n is number of features
        # jtdnn_obj is just an instance of the JTDNN class
        #print(jtdnn_obj)
        self.output_dims = input_dims # there is legit nothing happening, so I just made it so that it behaves as an identity function
        self.output_size = self.output_dims[0]
        
        self.jtdnn_obj = jtdnn_obj # creating an variable that is a reference to the jtdnn_obj object
        #print(self.jtdnn_obj)

class TestCases:

    def __init__(self):
        self.program_choose = 0
        self.choose_program(self.program_choose)
    
    def choose_program(self, program_choose):
    
        program_lis  = [self.test_adam,
                        self.test_relu,
                        self.load_and_test_mini_batches,
                        self.test_mini_batches,
                        self.test_import_model,
                        self.test_forward_prop,
                        self.test_fully_connected_NN, 
                        self.test_sigmoid_activation, 
                        self.test_linear_class, 
                        self.run_test_model]
                       
        program_lis[program_choose]()
    
    @staticmethod
    def test_adam():
        
        file_name = 'cubic_model.plk'
        basic_NN = JTDNN()
        input = basic_NN.input(input_dims = (2, None))
        Z1 = layers.Linear(output_dims = (10, None), initialiser = "glorot", name = "linear")(input) #10
        A1 = activations.Sigmoid(Z1, name = 'sigmoid')
        Z2 = layers.Linear(output_dims = (5, None), initialiser = "glorot", name = "linear")(A1) # 5
        A2 = activations.Sigmoid(Z2, name = 'sigmoid')
        Z3 = layers.Linear(output_dims = (1, None), initialiser = "glorot", name = "linear")(A2)
        output = activations.Sigmoid(Z3, name = 'sigmoid')
        
        #optimiser = optimisers.GradientDesc(learning_rate = 0.001)
        optimiser = optimisers.Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon=1e-07)
        fig_num_cost = 2
        
        loss = losses.BinaryCrossEntropy(basic_NN, store_cost = True, fig_num = fig_num_cost)

        basic_NN.compile(input = input, output = output, lambd = 0.01, loss = loss, metrics = "Accuracy", optimiser = optimiser) # BGD stands for Batch Gradient Descent # BGD stands for Batch Gradient Descent
        
        csv_file = r'C:\Users\josia\Desktop\Josiah_Folder\UNI\Semester_1\PEP1\robotics_club\YOLOv3_tiny\labelled_data2D_3.csv'

        
        plots = pd.read_csv(csv_file)
        plots = plots.to_numpy()
        
        X = plots[:,[0,1]].T
        
        X, mu, sigma = feature_norm(X)
        
        Y = plots[:,-1].astype(np.uint8).reshape(1,-1)
        
        fig_num_dec = 1
        mini_batch_size = 64
        num_epoches = 1000
        
        for epoch in range(num_epoches):
            mini_batch_num = 1
            for mini_batch_X, mini_batch_Y in mini_batch_generator(X, Y, mini_batch_size):
                
                AL = basic_NN.forward_prop(mini_batch_X)
                
                cost = basic_NN.compute_cost(mini_batch_Y)
                
                print('epoch %d accuracy after iteration %d: %4.2f' % (epoch, mini_batch_num, np.mean((AL >= 0.5) == mini_batch_Y) * 100))
                basic_NN.back_prop(mini_batch_Y)
                basic_NN.update_weights()
                mini_batch_num +=1
        

        plot_decision_boundary(X, Y, basic_NN, fig_num_dec)
        
        basic_NN.plot_cost(title = "Cost per Iteration", xlabel = "Number of iterations", ylabel = "Cost")
        
        
        
        plt.show()
    
    
    @staticmethod
    def test_relu():
        basic_NN = JTDNN()
        input = basic_NN.input(input_dims = (2, None))
        Z1 = layers.Linear(output_dims = (10, None), initialiser = "glorot", name = "linear")(input) #10
        A1 = activations.ReLu(Z1, name = 'relu')
        Z2 = layers.Linear(output_dims = (5, None), initialiser = "glorot", name = "linear")(A1) # 5
        A2 = activations.ReLu(Z2, name = 'relu')
        Z3 = layers.Linear(output_dims = (1, None), initialiser = "glorot", name = "linear")(A2)
        output = activations.Sigmoid(Z3, name = 'sigmoid')
        
        optimiser = optimisers.GradientDesc(learning_rate = 0.01)
        
        fig_num_cost = 2
        
        loss = losses.BinaryCrossEntropy(basic_NN, store_cost = True, fig_num = fig_num_cost)

        basic_NN.compile(input = input, output = output, lambd = 0.01, loss = loss, metrics = "Accuracy", optimiser = optimiser) # BGD stands for Batch Gradient Descent
        
        csv_file = r'C:\Users\josia\Desktop\Josiah_Folder\UNI\Semester_1\PEP1\robotics_club\YOLOv3_tiny\labelled_data2D_3.csv'
        
        
        plots = pd.read_csv(csv_file)
        plots = plots.to_numpy()
        
        X = plots[:,[0,1]].T
        
        X, mu, sigma = feature_norm(X)
        
        Y = plots[:,-1].astype(np.uint8).reshape(1,-1)
        
        fig_num_dec = 1
        
        
        for itera in range(1000000):
            AL = basic_NN.forward_prop(X)
            if itera % 10000 == 0:
                loss = basic_NN.compute_cost(Y)
                print(loss)
                #print('accuracy after iteration %d: %4.2f' % itera, np.mean((AL >= 0.5) == Y) * 100)
            basic_NN.back_prop(Y)
            
            basic_NN.update_weights()
        basic_NN.plot_cost(title = "Cost per Iteration", xlabel = "Number of number of iterations (10000s)", ylabel = "Cost")
        print(basic_NN.get_costs())
        plot_decision_boundary(X, Y, basic_NN, fig_num_dec)
        
        plt.show()
    
    @staticmethod
    def load_and_test_mini_batches():
        file_name = "cubic_model.plk"
        
        basic_NN = joblib.load(file_name)
        
        optimiser = optimisers.GradientDesc(learning_rate = 0.05)
        
        fig_num_cost = 2
        
        loss = losses.BinaryCrossEntropy(basic_NN, store_cost = True, fig_num = fig_num_cost)

        basic_NN.compile(lambd = 0.01, loss = loss, metrics = "Accuracy", optimiser = optimiser) # BGD stands for Batch Gradient Descent # BGD stands for Batch Gradient Descent
        
        csv_file = r'C:\Users\josia\Desktop\Josiah_Folder\UNI\Semester_1\PEP1\robotics_club\YOLOv3_tiny\labelled_data2D_3.csv'

        
        plots = pd.read_csv(csv_file)
        plots = plots.to_numpy()
        
        X = plots[:,[0,1]].T
        
        X, mu, sigma = feature_norm(X)
        
        Y = plots[:,-1].astype(np.uint8).reshape(1,-1)
        fig_num_dec = 1
        
        """
        mini_batch_size = 64
        num_epoches = 1000
        for epoch in range(num_epoches):
            mini_batch_num = 1
            for mini_batch_X, mini_batch_Y in mini_batch_generator(X, Y, mini_batch_size):
                AL = basic_NN.forward_prop(mini_batch_X)
                
                cost = basic_NN.compute_cost(mini_batch_Y)
                
                print('epoch %d accuracy after iteration %d: %4.2f' % (epoch, mini_batch_num, np.mean((AL >= 0.5) == mini_batch_Y) * 100))
                basic_NN.back_prop(mini_batch_Y)
                basic_NN.update_weights()
                mini_batch_num +=1
        """    
        
        for itera in range(1000000):
            AL = basic_NN.forward_prop(X)
            if itera % 10000 == 0:
                loss = basic_NN.compute_cost(Y)
                print(loss)
                #print('accuracy after iteration %d: %4.2f' % itera, np.mean((AL >= 0.5) == Y) * 100)
            basic_NN.back_prop(Y)
            
            basic_NN.update_weights()
        
        basic_NN.plot_cost(title = "Cost per Iteration", xlabel = "Number of iterations", ylabel = "Cost")
        
        plot_decision_boundary(X, Y, basic_NN, fig_num_dec)
        
        plt.show()
        
        #joblib.dump(basic_NN, file_name)
    @staticmethod
    def test_mini_batches():
        
        file_name = 'cubic_model.plk'
        basic_NN = JTDNN()
        input = basic_NN.input(input_dims = (2, None))
        Z1 = layers.Linear(output_dims = (10, None), initialiser = "glorot", name = "linear")(input) #10
        A1 = activations.Sigmoid(Z1, name = 'sigmoid')
        Z2 = layers.Linear(output_dims = (5, None), initialiser = "glorot", name = "linear")(A1) # 5
        A2 = activations.Sigmoid(Z2, name = 'sigmoid')
        Z3 = layers.Linear(output_dims = (1, None), initialiser = "glorot", name = "linear")(A2)
        output = activations.Sigmoid(Z3, name = 'sigmoid')
        
        optimiser = optimisers.GradientDesc(learning_rate = 0.001)
        
        fig_num_cost = 2
        
        loss = losses.BinaryCrossEntropy(basic_NN, store_cost = True, fig_num = fig_num_cost)

        basic_NN.compile(input = input, output = output, lambd = 0.01, loss = loss, metrics = "Accuracy", optimiser = optimiser) # BGD stands for Batch Gradient Descent # BGD stands for Batch Gradient Descent
        
        csv_file = r'C:\Users\josia\Desktop\Josiah_Folder\UNI\Semester_1\PEP1\robotics_club\YOLOv3_tiny\labelled_data2D_3.csv'

        
        plots = pd.read_csv(csv_file)
        plots = plots.to_numpy()
        
        X = plots[:,[0,1]].T
        
        X, mu, sigma = feature_norm(X)
        
        Y = plots[:,-1].astype(np.uint8).reshape(1,-1)
        
        fig_num_dec = 1
        mini_batch_size = 64
        num_epoches = 10
        """
        for epoch in range(num_epoches):
            for mini_batch in mini_batch_generator(X, Y, mini_batch_size):
                print ("shape of mini_batch_X: " + str(mini_batch[0].shape))
                print ("shape of mini_batch_Y: " + str(mini_batch[1].shape))
        """
        """
        shape of mini_batch_X: (2, 64)
        shape of mini_batch_Y: (1, 64)
        shape of mini_batch_X: (2, 64)
        shape of mini_batch_Y: (1, 64)
        shape of mini_batch_X: (2, 64)
        shape of mini_batch_Y: (1, 64)
        shape of mini_batch_X: (2, 7)
        shape of mini_batch_Y: (1, 7)
        """
        for epoch in range(num_epoches):
            mini_batch_num = 1
            for mini_batch_X, mini_batch_Y in mini_batch_generator(X, Y, mini_batch_size):
                """
                #random experiment here
                if mini_batch_X.shape[-1] != mini_batch_size:
                    print(mini_batch_X.shape)
                    continue
                """
                AL = basic_NN.forward_prop(mini_batch_X)
                
                cost = basic_NN.compute_cost(mini_batch_Y)
                
                print('epoch %d accuracy after iteration %d: %4.2f' % (epoch, mini_batch_num, np.mean((AL >= 0.5) == mini_batch_Y) * 100))
                basic_NN.back_prop(mini_batch_Y)
                basic_NN.update_weights()
                mini_batch_num +=1
            
        """
        for itera in range(1000000):
            AL = basic_NN.forward_prop(X)
            if itera % 10000 == 0:
                loss = basic_NN.compute_cost(Y)
                print(loss)
                #print('accuracy after iteration %d: %4.2f' % itera, np.mean((AL >= 0.5) == Y) * 100)
            basic_NN.back_prop(Y)
            
            basic_NN.update_weights()
        """
        basic_NN.plot_cost(title = "Cost per Iteration", xlabel = "Number of iterations", ylabel = "Cost")
        
        plot_decision_boundary(X, Y, basic_NN, fig_num_dec)
        
        plt.show()
    
    @staticmethod
    def test_import_model():
        file_name = "cubic_model.plk"
        basic_NN = joblib.load(file_name)
        csv_file = r'C:\Users\josia\Desktop\Josiah_Folder\UNI\Semester_1\PEP1\robotics_club\YOLOv3_tiny\labelled_data2D_3.csv'
        import pandas as pd
        from random_utils import feature_norm, plot_decision_boundary
        
        plots = pd.read_csv(csv_file)
        plots = plots.to_numpy()
        
        X = plots[:,[0,1]].T
        
        X, mu, sigma = feature_norm(X)
        
        Y = plots[:,-1].astype(np.uint8).reshape(1,-1)
        
        fig_num = 1
        
        
        AL = basic_NN.forward_prop(X)
            
        print(f'accuracy: {np.mean((AL >= 0.5) == Y) * 100}%')
            
            
        plot_decision_boundary(X, Y, basic_NN, fig_num)
        plt.show()
    @staticmethod
    def test_forward_prop():
        file_name = 'cubic_model.plk'
        basic_NN = JTDNN()
        input = basic_NN.input(input_dims = (2, None))
        Z1 = layers.Linear(output_dims = (10, None), initialiser = "glorot", name = "linear")(input) #10
        A1 = activations.Sigmoid(Z1, name = 'sigmoid')
        Z2 = layers.Linear(output_dims = (5, None), initialiser = "glorot", name = "linear")(A1) # 5
        A2 = activations.Sigmoid(Z2, name = 'sigmoid')
        Z3 = layers.Linear(output_dims = (1, None), initialiser = "glorot", name = "linear")(A2)
        output = activations.Sigmoid(Z3, name = 'sigmoid')
        
        optimiser = optimisers.GradientDesc(learning_rate = 0.01)
        
        fig_num_cost = 2
        
        loss = losses.BinaryCrossEntropy(basic_NN, store_cost = True, fig_num = fig_num_cost)

        basic_NN.compile(input = input, output = output, lambd = 0.01, loss = loss, metrics = "Accuracy", optimiser = optimiser) # BGD stands for Batch Gradient Descent
        
        csv_file = r'C:\Users\josia\Desktop\Josiah_Folder\UNI\Semester_1\PEP1\robotics_club\YOLOv3_tiny\labelled_data2D_3.csv'
        
        
        plots = pd.read_csv(csv_file)
        plots = plots.to_numpy()
        
        X = plots[:,[0,1]].T
        
        X, mu, sigma = feature_norm(X)
        
        Y = plots[:,-1].astype(np.uint8).reshape(1,-1)
        
        fig_num_dec = 1
        
        
        for itera in range(1000000):
            AL = basic_NN.forward_prop(X)
            if itera % 10000 == 0:
                loss = basic_NN.compute_cost(Y)
                print(loss)
                #print('accuracy after iteration %d: %4.2f' % itera, np.mean((AL >= 0.5) == Y) * 100)
            basic_NN.back_prop(Y)
            
            basic_NN.update_weights()
        basic_NN.plot_cost(title = "Cost per Iteration", xlabel = "Number of number of iterations (10000s)", ylabel = "Cost")
        print(basic_NN.get_costs())
        plot_decision_boundary(X, Y, basic_NN, fig_num_dec)
        
        plt.show()
        #joblib.dump(basic_NN, file_name) # testing whether we can dump the object in a file
        #print(A1, A2, output, Z1, Z2, Z3) # prints out all the objects
        """ sequence generated from print statements
        forward linear1
        forward sigmoid1
        forward linear2
        forward sigmoid2
        forward linear3
        forward sigmoid3
        50.25125628140703
        backward linear3
        backward sigmoid2
        backward linear2
        backward sigmoid1
        backward linear1
        update linear1
        update sigmoid1
        update linear2
        update sigmoid2
        update linear3
        update sigmoid3        
        """
        
    @staticmethod
    def test_fully_connected_NN():
        basic_NN = JTDNN()
        input = basic_NN.input(input_dims = (2, None))
        Z1 = layers.Linear(output_dims = (10, None), initialiser = "glorot", name = "linear")(input)
        A1 = activations.Sigmoid(Z1, name = 'sigmoid')
        Z2 = layers.Linear(output_dims = (5, None), initialiser = "glorot", name = "linear")(A1)
        A2 = activations.Sigmoid(Z2, name = 'sigmoid')
        Z3 = layers.Linear(output_dims = (1, None), initialiser = "glorot", name = "linear")(A2)
        output = activations.Sigmoid(Z3, name = 'sigmoid')
        print(f'basic_NN.graph_lis {basic_NN.graph_lis}') #['linear1']
        print(f'basic_NN.graph_dict {basic_NN.graph_dict}') # "linear1 <layers.linear object>
        print(f'output.jtdnn_obj {output.jtdnn_obj}')
        print(f'output.output_size {output.output_size}')
    @staticmethod
    def test_sigmoid_activation():
        basic_NN = JTDNN()
        input = basic_NN.input(input_dims = (2, None))
        Z1 = layers.Linear(output_dims = (10, None), initialiser = "glorot", name = "linear")(input)
        output = activations.Sigmoid(Z1, name = 'sigmoid')
        print(f'output.jtdnn_obj {output.jtdnn_obj}')
        print(f'output.output_size {output.output_size}')
        print(f'basic_NN.graph_lis {basic_NN.graph_lis}') #['linear1']
        print(f'basic_NN.graph_dict {basic_NN.graph_dict}') # "linear1 <layers.linear object>
        
    @staticmethod
    def test_linear_class():
        basic_NN = JTDNN()
        input = basic_NN.input(input_dims = (2, None))
        Z1 = layers.Linear(output_dims = (10, None), initialiser = "glorot", name = "linear")(input)
        print(f'basic_NN.graph_lis {basic_NN.graph_lis}') #['linear1']
        print(f'basic_NN.graph_dict {basic_NN.graph_dict}') # "linear1 <layers.linear object>
        print(f'input.jtdnn_obj {input.jtdnn_obj}') # JTDNN object
        print(f'input.output_dims {input.output_dims}') # (2, None)
        print(f'Z1 {Z1}')
        print(f'Z1.output_dims {Z1.output_dims}')
        print(f'Z1.output_size {Z1.output_size}')
        print(f'Z1.W {Z1.W}')
        print(f'Z1.b {Z1.b}')
        print(f'Z1.W.shape {Z1.W.shape}')
        print(f'Z1.b.shape {Z1.b.shape}')
        
        
    @staticmethod
    def run_test_model():
        basic_NN = JTDNN()
        input = basic_NN.input(input_dims = (2, None))
        Z1 = layers.Linear(output_dims = (10, None), initialiser = "glorot", name = "linear")(input)
        A1 = activations.Relu(Z1, name = 'relu')
        Z2 = layers.Linear(output_dims = (5, None), initialiser = "glorot", name = "Henry")(A1)
        A2 = activations.Relu(Z2, name = 'relu')
        Z3 = layers.Linear(output_dims = (1, None), initialiser = "glorot")(A2) #name shoud be automatically set to "Henry2"
        output = activations.Sigmoid(Z3, name = 'sigmoid')
        
        optimiser = optimisers.GradientDesc(learning_rate = 0.001)
        
        basic_NN.compile(input = input, output = output, lambd = 0.01, loss = "BinaryCrossEntropy", optimiser = optimiser) # BGD stands for Batch Gradient Descent
        
        #Basic_NN.fit(X, Y, num_iterations = 10000, verbose = 1)
        
        num_iterations = 10000
        for _ in range(num_iterations):
            basic_NN.forward_prop(X)
            
            basic_NN.compute_cost(Y)
            
            basic_NN.back_prop()
            
            basic_NN.update_weights()
    
    

if __name__ == '__main__':
    TestCases()
    
    