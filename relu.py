# write a class to implement ReLU activation function

import numpy as np

class ActivationLayer():
    def __init__(self):
        self.u = None
    
    def __str__(self):
        return 'ReLU'
    
    def forward(self, u):
        self.u = u
        v = np.copy(u)
        v[v < 0] = 0  # applying ReLU activation function
        return v
    
    def backward(self, del_v, lr):
        del_u = np.copy(self.u)
        del_u[del_u > 0] = 1  # applying sign(x) function for x > 0
        del_u[del_u < 0] = 0  # applying sign(x) function for x < 0
        del_u = del_v * del_u
        return del_u

    def set_learnable_parameters(self):
        pass