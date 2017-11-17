import numpy as np
import setup as su
import functions as f
import matplotlib.pyplot as plt


def forward_linear(A, W, b):
    
    #   Implement the linear part of a layer's forward propagation
    Z = np.dot(W, A) + b

    assert(Z.shape == (W.shape[0], A.shape[1]))
    
    return Z



def backward_linear(dZ, cache):
    A_prev, W, b = cache
    
    m = A_prev.shape[1]
    
    dW = np.dot(dZ, A_prev.T) / m
    
    db = np.sum(dZ, axis = 1, keepdims = True) / m
    
    dA_prev = np.dot(W.T, dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db



def backward_activation(dA, cache, func = f.relu_backwards):
    
    linear_cache, activation_cache = cache
    
    dZ = func(dA, activation_cache)
    dA_prev, dW, db = backward_linear(dZ, linear_cache)
    
    return dA_prev, dW, db


def model_forward(X, parameters):
    
    #   Implement forward propagation for LINEAR -> RELU * (L-1) -> LINEAR->SIGMOID
    caches = []    
    
    A = X

    layers = len(parameters) // 2
        
    for layer in range(1, layers):
        
        A_prev = A
        W = parameters['W' + str(layer)]
        b = parameters['b' + str(layer)]
                                             
        Z = forward_linear(A_prev, W, b)

        A = f.relu(Z)
        
        caches.append(((A_prev, W, b), Z))
        
    #   Execute final forward propagation which will end with Sigmoid function
    W = parameters['W' + str(layers)]
    b = parameters['b' + str(layers)]

    Z = forward_linear(A, W, b)

    caches.append(((A, W, b), Z))
    
    A = f.sigmoid(Z)

    return A, caches


def model_backward(AL, Y, caches):
    
    gradients = {}
    L = len(caches)

    Y = Y.reshape(AL.shape)
    
    #   Inititialize the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    #   L-th layer (Sigmoid -> Linear) gradients
    current_cache = caches[L - 1]
    
    gradients['dA' + str(L)], gradients['dW' + str(L)], gradients['db' + str(L)] = backward_activation(dAL, current_cache, f.sigmoid_backwards) 

    #   l-th layers (RELU -> Linear) gradients
    for layer in reversed(range(L - 1)):
        current_cache = caches[layer]

        dA_prev_temp, dW_temp, db_temp = backward_activation(gradients['dA' + str(layer + 2)], current_cache, f.relu_backwards)
        
        gradients['dA' + str(layer + 1)] = dA_prev_temp
        gradients['dW' + str(layer + 1)] = dW_temp
        gradients['db' + str(layer + 1)] = db_temp
    
    return gradients
        

def run_model(X, Y, layers_dims, learning_rate = .0075, num_iterations = 3000, print_cost = False):
    np.random.seed(1)
    costs = []

    parameters = su.initialize_parameters(layers_dims)
    
    for i in range(0, num_iterations):
        
        #   Forward propagation (Linear -> Relu * (L-1) -> Linear -> Sigmoid)
        AL, caches = model_forward(X, parameters)
            
        cost = f.compute_cost(AL, Y)
        
        gradients = model_backward(AL, Y, caches)
        
        parameters = su.update_parameters(parameters, gradients, learning_rate)
        
        #   Print the cost every 100 training examples
        if (print_cost and i % 100 == 0):
            print('Cost after iteration %i: %f' %(i, cost))
            
            costs.append(cost)
        
    #   Plot the cost
    plt.plot(np.squeeze(costs))  
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title('Learning rate = ' + str(learning_rate))
    plt.show()
    
    return parameters
