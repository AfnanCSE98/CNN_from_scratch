# write a class for implementing max pooling operation using 2 hyerparameters : filter_dim, stride

import numpy as np
import math 

class MaxPoolingLayer():
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.u_shape = None
        self.v_map = None
        self.verbose = False
    
    def __str__(self):
        return f'MaxPool(kernel={self.kernel_size}, stride={self.stride})'
    
    def forward(self, u):

        if self.verbose:
            print("start of maxPooling forward : " , u.shape)
        self.u_shape = u.shape
        
        num_samples, input_dim, _, num_channels = u.shape
        output_dim = math.floor((input_dim - self.kernel_size) / self.stride) + 1

        stride = self.stride
        kernel_size = self.kernel_size
        
        v = np.zeros((num_samples, output_dim, output_dim, num_channels))
        self.v_map = np.zeros((num_samples, output_dim, output_dim, num_channels)).astype(np.int32)
        
        
        vectorize = True
        if not vectorize:        
            for k in range(num_samples):
                for l in range(num_channels):
                    for i in range(output_dim):
                        for j in range(output_dim):
                            v[k, i, j, l] = np.max(u[k, i * self.stride: i * self.stride + self.kernel_size, j * self.stride: j * self.stride + self.kernel_size, l])
                            self.v_map[k, i, j, l] = np.argmax(u[k, i * self.stride: i * self.stride + self.kernel_size, j * self.stride: j * self.stride + self.kernel_size, l])
        else:
            strides = (stride* input_dim  ,stride, input_dim , 1)
            strides = tuple(i * u.itemsize for i in strides)

            subM = np.lib.stride_tricks.as_strided(u, shape=( output_dim , output_dim, kernel_size , kernel_size), strides=strides)

            v = np.zeros((num_samples, output_dim, output_dim, num_channels))

            for k in range(num_samples):
                for l in range(num_channels):
                    v[k,:,:,l] = np.max(subM, axis=(2,3))
                    
            # calculate v_map 
            for k in range(num_samples):
                for l in range(output_dim):
                    self.v_map[k,l,:,:] = np.argmax(subM[k,l,:,:])



        if self.verbose:
            print("end of maxPooling forward : " , v.shape)
        return v
    
    def backward(self, del_v, lr):
        del_u = np.zeros(self.u_shape)
        num_samples, input_dim, _, num_channels = del_v.shape
        
        for k in range(num_samples):
            for l in range(num_channels):
                for i in range(input_dim):
                    for j in range(input_dim):
                        position = tuple(sum(pos) for pos in zip((self.v_map[k, i, j, l] // self.kernel_size, self.v_map[k, i, j, l] % self.kernel_size), (i * self.stride, j * self.stride)))
                        del_u[(k,) + position + (l,)] = del_u[(k,) + position + (l,)] + del_v[k, i, j, l]
        
        
        # positions = tuple(sum(pos) for pos in zip((self.v_map // self.kernel_size, self.v_map % self.kernel_size), (np.indices((input_dim, input_dim)) * self.stride)))
        # del_u[np.arange(num_samples)[:, np.newaxis, np.newaxis, np.newaxis], positions[0], positions[1], positions[2]] += del_v
        
        return del_u

    def update_learnable_parameters(self, del_w, del_b, lr):
        pass
    
    def save_learnable_parameters(self):
        pass
    
    def set_learnable_parameters(self):
        pass