import numpy as np 
import math 



# -------------------------   vectorize maxPool --------------------------------------

"""
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

"""

# -------------------------   vectorize convolution --------------------------------------
"""
kernel_size = 5
stride = 1
padding = 2

num_samples = 16
input_dim = 180
num_channels = 6

u = np.random.rand(num_samples, input_dim, input_dim, num_channels)

print("start of convolution forward : " , u.shape)
num_samples, input_dim, _, num_channels = u.shape

output_dim = math.floor((input_dim - kernel_size + 2 * padding) / stride) + 1

weights = None
biases = None

num_filters = 7

if weights is None:
    weights = np.random.randn(  num_filters , kernel_size, kernel_size, num_channels ) * math.sqrt(2 / (kernel_size * kernel_size * num_channels))
if biases is None:
    biases = np.zeros(num_filters)

u_pad = np.pad(u, ((0,), (padding,), (padding,), (0,)), mode='constant')
v = np.zeros((num_samples, output_dim, output_dim, num_filters))


strides = (stride* input_dim  ,stride, input_dim , 1)
strides = tuple(i * u.itemsize for i in strides)

subM = np.lib.stride_tricks.as_strided(u_pad, shape=( output_dim , output_dim, kernel_size , kernel_size), strides=strides)
print("subM shape : " , subM.shape , "weights shape : " , weights.shape , "biases shape : " , biases.shape)

for k in range(num_samples):
    for l in range(num_filters):
        tmp = np.einsum('ijkl,klp->ijp', subM, weights[l])
        v[k,:,:,l] = np.sum(tmp, axis=(1,2)) + biases[l]

print("end of convolution forward : " , v.shape)

# print("end of convolution forward : " , v.shape)

"""