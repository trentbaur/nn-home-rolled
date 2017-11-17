import numpy as np


#---------------------------
#   Forward Prop Functions
#---------------------------
def sigmoid(Z):
    
    A = 1 / (1 + np.exp(-Z))
        
    return A

# sigmoid(0.01090232)


def relu(Z):
    A = np.maximum(0, Z)
    
    return A


#---------------------------
#   Backprop Functions
#---------------------------

def sigmoid_backwards(dA, cache):
    
    Z = cache
    
    s = 1 / (1 + np.exp(-Z))
    
    dZ = dA * s * (1 - s)
    
    return dZ


def relu_backwards(dA, cache):

    Z = cache
    
    dZ = np.array(dA, copy = True)
        
    #   Set any negative values to zero
    dZ[Z <= 0] = 0
    
    return dZ


#---------------------------
#   Cost Functions
#---------------------------
def compute_cost(aL, Y):
    
    m = Y.shape[1]
    
    #   Compute loss from aL and Y
    cost = -np.sum(np.multiply(np.log(aL), Y) + np.multiply(np.log(1 - aL), (1 - Y))) / m
    
    #   Make sure cost shape is correct
    cost = np.squeeze(cost)
    
    return cost

