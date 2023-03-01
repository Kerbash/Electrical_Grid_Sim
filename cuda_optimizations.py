# ---------------INSTALLATION INSTRUCTIONS ----------------------------#
# https://numba.readthedocs.io/en/stable/cuda/overview.html
# Use the following command if using Conda:
# $ conda install cudatoolkit

# ---------------LIBRARY USED -----------------------------------------#
# Info about the numba library: https://numba.readthedocs.io/en/stable/cuda/index.html
# Numba does not implement all features of CUDA, yet. 
# Some missing features are listed below:
# dynamic parallelism
# texture memory

# Learn more about our GPU using this function:
cuda.detect()

from numba import cuda
import math

# Example of a kernel function -- a square root function from https://thedatafrog.com/en/articles/cuda-kernel-python/
@cuda.jit
def kernel(x, out):
  idx = cuda.grid(1)
  out[idx] = math.sqrt(x[idx])

# Example of a strided kernel
@cuda.jit
def kernel(x, out):
  start = cuda.grid(1)      #
  stride = cuda.gridsize(1) # the stride is the same as the width of a block (gridsize)
  for i in range(start, x.shape[0], stride): 
    out[i] = math.atan(x[i])



import numpy as np

# Adapted from https://colab.research.google.com/github/cbernet/maldives/blob/master/numba/numba_cuda_kernel.ipynb#scrollTo=fACSmHLzJanZ

A = np.arange(4096,dtype=np.float32)

# We can time our kernels using this:
# %timeit (put a function call next to it)

# Allocate space on the device for the input data
d_A = cuda.to_device(A)

# Output data - the GPU will fill this out:
d_Out = cuda.device_array_like(d_A)

# <<< 32, 128 >>> configuration
blocks_per_grid = 32    # gridsize
threads_per_block = 128 # blocksize
kernel[blocks_per_grid, threads_per_block](d_A, d_Out)

# wait for all threads to complete
cuda.synchronize()

# Return the matrix to the host
d_Out.copy_to_host()


# https://thedatafrog.com/en/articles/boost-python-gpu/
# We may have to store things in contiguous memory. Example:
# x = np.ascontiguousarray(points[:,0])
# y = np.ascontiguousarray(points[:,1])


# We may need to make our functions into ufunc
# What is a ufunc?
# It is a numpy universal function that operate element-by-element on a numpy array
# ufuncs are implemented using C -- that's why they are so fast
# "create and compile ufuncs for the GPU to perform the calculation on many elements at the same time."

# Creates a ufunc to enable parallelizing threads:
# @vectorize(['float32(float32)'], target='cuda')
# def gpu_sqrt(x):
#     return math.sqrt(x)

# # More than one input value:
# @vectorize(['float32(float32, float32)'],target='cuda')
# def gpu_arctan2(y, x): 
#     theta = math.atan2(y,x)
#     return theta
