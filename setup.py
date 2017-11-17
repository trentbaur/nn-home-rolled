import numpy as np

def initialize_parameters(layer_dims = [12288, 20, 7, 1]):
    
    np.random.seed(1)
    
    parameters = {}
    
    L = len(layer_dims)
    
    #   Set parameters for each layer to non-zero values
    for layer in range(1, L):
        parameters['W' + str(layer)] = np.random.randn(layer_dims[layer], layer_dims[layer - 1]) / np.sqrt(layer_dims[layer - 1]) #* 0.01
        parameters['b' + str(layer)] = np.zeros((layer_dims[layer], 1))
    
    return parameters
        
#   initialize_parameters()
#   initialize_parameters(layers_dims)



def update_parameters(parameters, gradients, learning_rate):
    
    L = len(parameters) // 2
    
    #   Update rule for each parameter
    for layer in reversed(range(1, L + 1)):
        parameters['W' + str(layer)] = parameters['W' + str(layer)] - (learning_rate * gradients['dW' + str(layer)])
        parameters['b' + str(layer)] = parameters['b' + str(layer)] - (learning_rate * gradients['db' + str(layer)])
        
    return parameters

