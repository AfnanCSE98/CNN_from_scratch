import numpy as np 
import math 



# -------------------------   vectorize maxPool --------------------------------------


kernel_size = 2
stride = 2

num_samples = 1
input_dim = 6
num_channels = 2

u = np.random.rand(num_samples, input_dim, input_dim, num_channels)
print(u.shape)
output_dim = math.floor((input_dim - kernel_size) / stride) + 1


strides = (stride* input_dim  ,stride, input_dim , 1)
strides = tuple(i * u.itemsize for i in strides)

subM = np.lib.stride_tricks.as_strided(u, shape=( output_dim , output_dim, kernel_size , kernel_size), strides=strides)

v = np.zeros((num_samples, output_dim, output_dim, num_channels))
v_map = np.zeros((num_samples, output_dim, output_dim, num_channels)).astype(np.int32)

v1_map = v_map 
v1 = v 

for k in range(num_samples):
    for l in range(num_channels):
        for i in range(output_dim):
            for j in range(output_dim):
                v1[k, i, j, l] = np.max(u[k, i * stride: i * stride + kernel_size, j * stride: j * stride + kernel_size, l])
                # print(u[k, i * stride: i * stride + kernel_size, j * stride: j * stride + kernel_size, l].shape)
                v1_map[k, i, j, l] = np.argmax(u[k, i * stride: i * stride + kernel_size, j * stride: j * stride + kernel_size, l])

for k in range(num_samples):
    for l in range(num_channels):
        v[k,:,:,l] = np.max(subM, axis=(2,3))
        
# calculate v_map 
for k in range(num_samples):
    for l in range(output_dim):
        v_map[k,l,:,:] = np.argmax(subM[k,l,:,:])


# for k in range(num_samples):
#     for l in range(num_channels):
#         subM_ = subM[k,:,:,l]
#         v[k,:,:,l] = np.max(subM, axis=(2,3))
#         v_map_flat = np.argmax(subM_, axis=(2,3))
#         v_map_indices = np.unravel_index(v_map_flat, subM_[::stride,::stride].shape)
#         v_map[k,:,:,l] = (v_map_indices[0]//stride, v_map_indices[1]//stride)


# v = np.amax(subM, axis=(2,3))
# v = np.expand_dims(v,axis=0)
# v = np.repeat(v, num_samples, axis=0)
# v = np.expand_dims(v, axis=3)
# v = np.repeat(v, num_channels, axis=3)

print("v :" , v.shape , "v1 :" , v1.shape)

print(v)
print("-----------------")
print(v1)
print("v1 nad v are same? " , np.allclose(v1, v))

print("v_map :and v1_map are same? " , np.allclose(v1_map, v_map))

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

v1 = v
for k in range(num_samples):
    for l in range(num_filters):
        for i in range(output_dim):
            for j in range(output_dim):
                v1[k, i, j, l] = np.sum(u_pad[k, i * stride: i * stride + kernel_size, j * stride: j * stride + kernel_size, :] * weights[l]) + biases[l]


strides = (stride* input_dim  ,stride, input_dim , 1)
strides = tuple(i * u.itemsize for i in strides)

subM = np.lib.stride_tricks.as_strided(u_pad, shape=( output_dim , output_dim, kernel_size , kernel_size), strides=strides)
print("subM shape : " , subM.shape , "weights shape : " , weights.shape , "biases shape : " , biases.shape)

for k in range(num_samples):
    for l in range(num_filters):
        tmp = np.einsum('ijkl,klp->ijp', subM, weights[l])
        v[k,:,:,l] = np.sum(tmp, axis=(1,2)) + biases[l]

print("end of convolution forward : " , v.shape)

# check if v1 and v are the same
print(np.allclose(v1, v))

"""


