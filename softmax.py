# write a class to implement softmax activation function

import numpy as np
class SoftmaxLayer():
    def __init__(self):
        pass
    
    def __str__(self):
        return 'Softmax'
    
    def forward(self, u):
        u = u - np.max(u, axis=0, keepdims=True)
        v = np.exp(u)
        v = v / np.sum(v, axis=0)
        return v
    
    def backward(self, del_v, lr):
        del_u = np.copy(del_v)
        return del_u

    def update_learnable_parameters(self, del_w, del_b, lr):
        pass
    
    def save_learnable_parameters(self):
        pass
    
    def set_learnable_parameters(self):
        pass