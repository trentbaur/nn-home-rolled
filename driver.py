import numpy as np
import setup

def model_forward(X, parameters):
    
    #   Implement forward propagation for LINEAR -> RELU * (L-1) -> LINEAR->SIGMOID
    caches = []
    
    A = X
    
    layers = len(parameters) // 2
    
    for layer in range(1, layers):
        A_prev = A
        
        A, cache = linear_activation_forward(A_prev,
                                             parameters['W' + str(layer)],
                                             parameters['b' + str(layer)])
        
        caches.append(cache)
        
    #   Execute final forward propagation which will end with Sigmoid function
    A, cache = linear_activation_forward(A,
                                         parameters['W' + str(layers)],
                                         parameters['b' + str(layers)],
                                         func = act.sigmoid)
    
    caches.append(cache)
    
    return A, caches

    