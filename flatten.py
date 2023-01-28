# write a class to implement Flattening layer: it will convert a (series of) convolutional filter maps to a column vector

import numpy as np
class FlatteningLayer():
    def __init__(self):
        self.u_shape = None
    
    def __str__(self):
        return 'Flatten'
    
    def forward(self, u):
        self.u_shape = u.shape
        v = np.copy(u)
        v = np.reshape(v, (v.shape[0], np.prod(v.shape[1:])))
        v = np.transpose(v)
        return v
    
    def backward(self, del_v, lr):
        del_u = np.copy(del_v)
        del_u = np.transpose(del_u)
        del_u = np.reshape(del_u, self.u_shape)
        return del_u

    def update_learnable_parameters(self, del_w, del_b, lr):
        pass
    
    def save_learnable_parameters(self):
        pass
    
    def set_learnable_parameters(self):
        pass