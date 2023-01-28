# write a class to implement softmax activation function

import numpy as np
class SoftmaxLayer():
    def __init__(self):
        pass
    
    def __str__(self):
        return 'Softmax'
    
    def forward(self, u):
        v = np.exp(u)
        v = v / np.sum(v, axis=0)
        return v
    
    def backward(self, del_v, lr):
        del_u = np.copy(del_v)
        return del_u

    def set_learnable_parameters(self):
        pass