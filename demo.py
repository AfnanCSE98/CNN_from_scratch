import numpy as np 
import math 



# -------------------------   vectorize maxPool --------------------------------------

"""
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



print("v1 nad v are same? " , np.allclose(v1, v))

print("v_map :and v1_map are same? " , np.allclose(v1_map, v_map))
"""
# -------------------------   vectorize convolution --------------------------------------

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
# for k in range(num_samples):
#     for l in range(num_filters):
#         for i in range(output_dim):
#             for j in range(output_dim):
#                 v1[k, i, j, l] = np.sum(u_pad[k, i * stride: i * stride + kernel_size, j * stride: j * stride + kernel_size, :] * weights[l]) + biases[l]


print((stride* input_dim  ,stride, num_filters, input_dim , 1))
strides = (stride* input_dim  ,stride, num_filters , input_dim , 1)
strides = tuple(i * u.itemsize for i in strides)

print("strides : " , strides)

subM = np.lib.stride_tricks.as_strided(u_pad, shape=(output_dim , output_dim, num_filters, kernel_size , kernel_size), strides=strides)

print("subM shape : " , subM.shape , "weights shape : " , weights.shape , "biases shape : " , biases.shape)
print("output_dim : " , output_dim , "kernel_size : " , kernel_size , "stride : " , stride , "padding : " , padding)

# correct

tmp = np.einsum('ijpkl,pklf->ijf', subM, weights)
v= np.sum(tmp, axis=(0,1)) + biases
print("end of convolution forward : " , v.shape)

# check if v1 and v are the same 
print(np.allclose(v1, v))




# -------------------------   vectorize convolution backward--------------------------------------
kernel_size = 5
stride = 1
padding = 2

num_samples = 16
input_dim = 180
num_channels = 6

u = np.random.rand(num_samples, input_dim, input_dim, num_channels)
u_pad = np.pad(u, ((0,), (padding,), (padding,), (0,)), mode='constant')

weights = None
biases = None

num_filters = 7

if weights is None:
    weights = np.random.randn(  num_filters , kernel_size, kernel_size, num_channels ) * math.sqrt(2 / (kernel_size * kernel_size * num_channels))
if biases is None:
    biases = np.zeros(num_filters)

del_v = np.random.rand(16, 39, 39, 100)

num_samples = del_v.shape[0]
input_dim = del_v.shape[1]
input_dim_pad = (input_dim - 1) * stride + 1
output_dim = u_pad.shape[1] - 2 * padding
num_channels = u_pad.shape[3]

del_b = np.sum(del_v, axis=(0, 1, 2)) / num_samples
del_v_sparse = np.zeros((num_samples, input_dim_pad, input_dim_pad, num_filters))
del_v_sparse[:, :: stride, :: stride, :] = del_v
weights_prime = np.rot90(np.transpose(weights, (3, 1, 2, 0)), 2, axes=(1, 2))
del_w = np.zeros((num_filters, kernel_size, kernel_size, num_channels))

for l in range(num_filters):
    for i in range(kernel_size):
        for j in range(kernel_size):
            del_w[l, i, j, :] = np.mean(np.sum(u_pad[:, i: i + input_dim_pad, j: j + input_dim_pad, :] * np.reshape(del_v_sparse[:, :, :, l], del_v_sparse.shape[: 3] + (1,)), axis=(1, 2)), axis=0)
      
# vectorize del_w 
del_w1 = np.zeros((num_filters, kernel_size, kernel_size, num_channels))
strides = (stride* input_dim_pad  ,stride, input_dim_pad , 1)
strides = tuple(i * u_pad.itemsize for i in strides)

subM = np.lib.stride_tricks.as_strided(u_pad, shape=( output_dim , output_dim, kernel_size , kernel_size), strides=strides)

for k in range(num_filters):
    del_w1[k] = np.einsum('ijkl,ijkl->klp', subM, del_v_sparse[:, :, :, k]) / num_samples

print("del_w and del_w1 are the same? " , np.allclose(del_w, del_w1))


