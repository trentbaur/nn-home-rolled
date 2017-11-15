import numpy as np

def sigmoid(Z):
    
    A = 1 / (1 + np.exp(-Z))
    
    cache = Z
    
    return A, cache


def relu(Z):
    A = np.maximum(0, Z)
    
    cache = Z
    
    return A, cache



def sigmoid_backwards(dA, cache):
    
    Z = cache
    
    s = 1 / (1 + np.exp(-Z))
    
    dZ = dA * s * (1 - s)
    
    return dZ


def relu_backwards(dA, cache)

    Z = cache
    
    dZ = np.array(dA, copy = True)
        
    #   Set any negative values to zero
    dZ[Z <= 0] = 0
    
    return dZ
