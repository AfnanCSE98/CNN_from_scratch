# write a class to implement Fully-connected layer: a dense layer. There will be one parameter: output dimension.

import numpy as np
import math 

class FullyConnectedLayer():
    def __init__(self, output_dim):
        self.output_dim = output_dim
        self.weights = None
        self.biases = None
        self.weights_matrix = None
        self.biases_vector = None
        self.u = None
    
    def __str__(self):
        return f'FullyConnected(output_dim={self.output_dim})'
    
    def forward(self, u):
        self.u = u
        
        if self.weights is None:
            self.weights = np.random.randn(self.output_dim, u.shape[0]) * math.sqrt(2 / u.shape[0])
        if self.biases is None:
            self.biases = np.zeros((self.output_dim, 1))
        
        v = self.weights @ u + self.biases
        return v
    
    def backward(self, del_v, lr):
        del_w = (del_v @ np.transpose(self.u)) / del_v.shape[1]
        del_b = np.reshape(np.mean(del_v, axis=1), (del_v.shape[0], 1))
        del_u = np.transpose(self.weights) @ del_v
        self.update_learnable_parameters(del_w, del_b, lr)
        return del_u
    
    def update_learnable_parameters(self, del_w, del_b, lr):
        self.weights = self.weights - lr * del_w
        self.biases = self.biases - lr * del_b
    
    def save_learnable_parameters(self):
        self.weights_matrix = np.copy(self.weights)
        self.biases_vector = np.copy(self.biases)
    
    def set_learnable_parameters(self):
        self.weights = self.weights if self.weights_matrix is None else np.copy(self.weights_matrix)
        self.biases = self.biases if self.biases_vector is None else np.copy(self.biases_vector)