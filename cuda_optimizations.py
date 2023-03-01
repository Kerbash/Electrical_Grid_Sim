#---------------------------------------------------------------------------
# CUDA_OPTIMIZATIONS.PY
# Using the GPU to accelerate an electrical power grid visualization
# Date:                   03/01/2023
# Authors:                Pragati Dode, Breanna Powell, and William Selke
# 
# +++++++++++++++++ DETAILS ABOUT SYSTEM ++++++++++++++
# IDE:                    Visual Studio Code
# Hosts Used:             ____
# CUDA Version:           _____
# Device Architecture:    Ampere
#
# +++++++++++++++++ INSTALLATION INSTRUCTIONS +++++++++++++++++ 
# https://numba.readthedocs.io/en/stable/cuda/overview.html
# Use the following command if using Conda:
# $ conda install cudatoolkit

# +++++++++++++++++ LIBRARY USED +++++++++++++++++ 
# Numba library information: https://numba.readthedocs.io/en/stable/cuda/index.html
# Note: Numba does not implement: dynamic parallelism and texture memory

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
  start = cuda.grid(1)      # This tells to start with the leftmost index within the block
  stride = cuda.gridsize(1) # The stride is the same as the width of a block (gridsize)
  for i in range(start, x.shape[0], stride): # Shape gives the size of the input array
    out[i] = math.atan(x[i])



import numpy as np
import time

# Adapted from https://colab.research.google.com/github/cbernet/maldives/blob/master/numba/numba_cuda_kernel.ipynb#scrollTo=fACSmHLzJanZ
A = np.arange(4096,dtype=np.float32) #Creates an array of size 4096 of floats

# Allocate space on the device for the input data:
d_A = cuda.to_device(A)

# Allocate space on the device for the output data:
d_Out = cuda.device_array_like(d_A)

# <<< 32, 128 >>> configuration
blocks_per_grid = 32    # gridsize
threads_per_block = 128 # blocksize

# We can time our kernels using this:
# https://realpython.com/python-timer/
startTimer = time.perf_counter()

# +++++++++++++++++ KERNEL FUNCTION +++++++++++++++++ 
kernel[blocks_per_grid, threads_per_block](d_A, d_Out)

# wait for all threads to complete
cuda.synchronize()
stopCompTime = time.perf_counter()

# Return the matrix to the host
d_Out.copy_to_host()

stopFullTime = time.perf_counter()

# See timing results
print(f"Time spent just computing : {stopCompTime - startTimer:0.8f} seconds")
print(f"Time with memory copying  : {stopFullTime - startTimer:0.8f} seconds")



# +++++++++++++++++ MORE NOTES +++++++++++++++++ 

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
