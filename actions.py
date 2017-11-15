import numpy as np
import functions as func

def forward_linear(A, W, b):
    
    #   Implement the linear part of a layer's forward propagation
    Z = np.dot(W, A) + b
    
    cache = (A, W, b)
    
    return Z, cache


def forward_activation(A_prev, W, b,
                       function = func.relu):
    
    #   Implement the forward activation function
    Z, linear_cache = forward_linear(A_prev, W, b)
    
    A, activation_cache = function(Z)
    
    cache = (linear_cache, activation_cache)
    
    return A, cache

