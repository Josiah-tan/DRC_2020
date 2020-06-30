
#================================================================
#
#   File name   : random_utils.py
#   Author      : Josiah Tan
#   Created date: 23/06/2020
#   Description : Contains functions, but not sure where to put them tbh
#
#================================================================

#================================================================

import numpy as np
import matplotlib.pyplot as plt
from math import floor
def feature_norm(X):
           
    m = X.shape[1]
    
    mu = np.mean(X, axis = 1, keepdims = True)
    
    
    sigma = np.std(X, axis = 1, keepdims = True)
                
    X_norm = np.divide((X - mu), sigma)
    
    return X_norm, mu, sigma
    
    
    
def plot_decision_boundary(X, Y, jtdnn_obj, fig_num):
    plt.figure(fig_num)
    
    X1 = X[0,:]
    X2 = X[1,:]
    
    
    
    
    assert X.shape[0] == 2
    
    #print(parameters)
    #assert isinstance(jtdnn_obj, JTDNN)
    
    
    
    num_val = 100
    
    x1 = np.linspace(np.min(X1), np.max(X1), num = num_val)
    x2 = np.linspace(np.min(X2), np.max(X2), num = num_val)
    
    xx, yy = np.meshgrid(x1, x2);
    
    X = np.concatenate((xx.reshape(1, -1), yy.reshape(1,-1)), axis = 0)
    
    AL = jtdnn_obj.forward_prop(X)
    

    cs = plt.contourf(xx, yy, AL.reshape(xx.shape), cmap=plt.cm.Paired, levels = [0, 0.5, 1])
    plt.colorbar(cs)
    
    LABEL_COLOR_MAP = {0 : 'r',
                       1 : 'k'
                       }
    labels = Y.tolist()
    label_color = [LABEL_COLOR_MAP[l] for l in labels[0]]
    
    
    plt.scatter(X1, X2, c = label_color)
    
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    
    Note:
    Shuffling and Partitioning are the two steps required to build mini-batches
    Powers of two are often chosen to be the mini-batch size, e.g., 16, 32, 64, 128.
    """
    
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size : (k+1)*mini_batch_size]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, num_complete_minibatches*mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches*mini_batch_size:]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
"""
if __name__ == "__main__":
    np.random.seed(1)
    mini_batch_size = 64
    X = np.random.randn(12288, 148)
    Y = np.random.randn(1, 148) < 0.5
    mini_batches = random_mini_batches(X, Y, mini_batch_size)

    print ("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
    print ("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
    print ("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))
    print ("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
    print ("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape)) 
    print ("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))
    print ("mini batch sanity check: " + str(mini_batches[0][0][0][0:3]))
"""   
    
def mini_batch_generator(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    yields:
    mini_batche -- tuple of synchronous (mini_batch_X, mini_batch_Y)
    
    Note:
    Shuffling and Partitioning are the two steps required to build mini-batches
    Powers of two are often chosen to be the mini-batch size, e.g., 16, 32, 64, 128.
    """
    
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    
    assert Y.shape[1] == m
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size : (k+1)*mini_batch_size]
        ### END CODE HERE ###
        yield (mini_batch_X, mini_batch_Y)
        
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, num_complete_minibatches*mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches*mini_batch_size:]
        ### END CODE HERE ###
        yield (mini_batch_X, mini_batch_Y)
         
if __name__ == "__main__":
    np.random.seed(1)
    mini_batch_size = 64
    X = np.random.randn(12288, 148)
    Y = np.random.randn(1, 148) < 0.5
    for mini_batch in mini_batch_generator(X, Y, mini_batch_size):
    
        print ("shape of mini_batch_X: " + str(mini_batch[0].shape))
        print ("shape of mini_batch_Y: " + str(mini_batch[1].shape))
        #print (f"mini_batch_X {mini_batch[0]}")
        #print (f"mini_batch_Y {mini_batch[1]}")

def image_resize(raw_image, dimensions, method = "nearest neighbour"):
    """
    parameters -- raw_image - a numpy array of size y, x, n or y,x
               -- dimensions - a tuple containing the resized dimensions
                             - example: (416, 416, 3) or (150, 150)
               -- method - method of resizing
    """
    resized_image = np.array(Image.fromarray(raw_image).resize((dimensions[1],dimensions[0]))) # resize image - nearest neighbour technique
    return resized_image

def flip_horizontal(x):
    # x.shape = (height, width, channels) or (height, width)
    assert len(x.shape) >= 2
    #print(x.shape)
    x_inv = x[:,::-1]
    #print(x_inv.shape)
    assert x.shape == x_inv.shape
    return x_inv
    

def flip_left_right(y):
    # sample: y = np.array([1,0,1,0])
    if np.array_equal(y, np.array([1,0,1,0])):
        return np.array([0,1,1,0])
    elif np.array_equal(y, np.array([0,1,1,0])):
        return np.array([1,0,1,0])

def sim_data_inversion_Y(X, Y):
    X_inv = []
    Y_inv = []
    assert X.shape[0] == Y.shape[0]
    assert Y.shape[1] == 4
    count = []
    for img_index in range(Y.shape[0]):
        x = X[img_index]
        
        y = Y[img_index]
        y_inv = flip_left_right(y)
        if y_inv is None:
            continue
        x_inv = flip_horizontal(x)
        X_inv.append(x_inv)
        Y_inv.append(y_inv)
        count.append(img_index)
    return (np.array(X_inv), np.array(Y_inv), count)
    
if __name__ == "__main__":
    from PIL import Image
    X1 = np.expand_dims(image_resize(plt.imread("aphid_train.jpg"),(100,100)), axis = 0)
    X2 = np.expand_dims(image_resize(plt.imread("aphid_backyard.jpg"),(100,100)), axis = 0)
    X3 = np.expand_dims(image_resize(plt.imread("aphid_test.jpg"),(100,100)), axis = 0)
    X4 = np.expand_dims(image_resize(plt.imread("aphid_test_2.jpg"),(100,100)), axis = 0)
    X = np.concatenate((X1, X2, X3, X4), axis = 0)
    Y = np.array([[1,0,1,0], [0,1,1,0], [0,1,1,0],[1,0,1,0]])
    X_inv, Y_inv = sim_data_inversion_Y(X, Y)
    for i in range(Y_inv.shape[0]):
        plt.subplot(1,4,i + 1)
        plt.imshow(X_inv[i])
        
    plt.show()