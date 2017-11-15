import numpy as np

def initialize_parameters(layer_dims = [12288, 20, 7, 1]):
    
    np.random.seed(1)
    
    parameters = {}
    
    L = len(layer_dims)
    
    #   Set parameters for each layer to non-zero values
    for layer in range(1, L):
        parameters['W' + str(layer)] = np.random.randn(layer_dims[layer], layer_dims[layer - 1]) * 0.01
        parameters['b' + str(layer)] = np.random.randn(layer_dims[layer], 1)
    
    return parameters
        
#   initialize_parameters()
