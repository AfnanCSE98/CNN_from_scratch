import numpy as np 
import math 

kernel_size = 2
stride = 2

num_samples = 16
input_dim = 180
num_channels = 6

u = np.random.rand(num_samples, input_dim, input_dim, num_channels)

output_dim = math.floor((input_dim - kernel_size) / stride) + 1


strides = (stride* input_dim  ,stride, input_dim , 1)
strides = tuple(i * u.itemsize for i in strides)

subM = np.lib.stride_tricks.as_strided(u, shape=( output_dim , output_dim, kernel_size , kernel_size), strides=strides)

v = np.zeros((num_samples, output_dim, output_dim, num_channels))

# for k in range(num_samples):
#     for l in range(num_channels):
#         v[k,:,:,l] = np.max(subM, axis=(2,3))

v = np.amax(subM, axis=(2,3))
v = np.expand_dims(v,axis=0)
v = np.repeat(v, num_samples, axis=0)
v = np.expand_dims(v, axis=3)
v = np.repeat(v, num_channels, axis=3)
print(v.shape)

