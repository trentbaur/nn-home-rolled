import numpy as np
import h5py


def load_data(type = 'train',
              file = 'catvnoncat'):
    
    dataset = h5py.File(name = 'datasets/' + type + '_' + file + '.h5', mode = 'r')
    set_x_orig = np.array(dataset[type + '_set_x'][:])
    set_y_orig = np.array(dataset[type + '_set_y'][:])
    set_y_orig = set_y_orig.reshape((1, set_y_orig.shape[0]))
    
    classes = np.array(dataset['list_classes'[:]])
    
    return set_x_orig, set_y_orig, classes
    

def load_dataset():
    
    train_X, train_y, classes = load_data(type = 'train', file = 'catvnoncat')
    test_X, test_y, _ = load_data(type = 'test', file = 'catvnoncat')
    
    return train_X, train_y, test_X, test_y, classes

#   load_dataset()