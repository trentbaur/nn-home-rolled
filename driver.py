import numpy as np
import load_data as dt
import actions

np.random.seed(1)

layers_dims = [12288, 20, 5, 1]


train_x_orig, train_y, test_x_orig, test_y, classes = dt.load_dataset()

#   Reshape the training and test examples
train_X = train_x_orig.reshape(train_x_orig.shape[0], -1).T / 255
test_X = test_x_orig.reshape(test_x_orig.shape[0], -1).T / 255


params = actions.run_model(train_X, train_y, layers_dims, num_iterations = 2000, print_cost = True)

