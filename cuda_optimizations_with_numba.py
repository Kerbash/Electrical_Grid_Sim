#---------------------------------------------------------------------------
# CUDA_OPTIMIZATIONS_WITH_NUMBA.PY
# Using the GPU to accelerate an electrical power grid visualization
#
# Date:                   03/14/2023
# Authors:                Pragati Dode, Breanna Powell, and William Selke
# 
# +++++++++++++++++ DETAILS ABOUT SYSTEM ++++++++++++++
# IDEs:                   Visual Studio Code; PyCharm
# Host Used:              ____put Will's computer info here
# Device Used:            ____put Will's computer info here
# CUDA Version:           _____put Will's computer info here
# Device Architecture:    Ampere
#
# +++++++++++++++++ INSTALLATION INSTRUCTIONS +++++++++++++++++
# https://numba.readthedocs.io/en/stable/user/installing.html
#
# Use the following commands if using Conda:
# $ conda install numba
# $ conda install cudatoolkit
#
# +++++++++++++++++ LIBRARY USED +++++++++++++++++ 
# Numba library information: https://numba.readthedocs.io/en/stable/cuda/overview.html
# Numba library contents: https://numba.readthedocs.io/en/stable/cuda/index.html
# Note: Numba does not implement: dynamic parallelism and texture memory

from numba import cuda
import math    
import numpy as np
import time
import sys

TOTAL_HOUSES = 320000
MAX_BLOCKS = 65535

# Learn more about our GPU using this function:
# cuda.detect()

# --------------------------------- KERNEL FUNCTION --------------------------------------
# See example kernel functions: 
# https://thedatafrog.com/en/articles/cuda-kernel-python/
# https://colab.research.google.com/github/cbernet/maldives/blob/master/numba/numba_cuda_kernel.ipynb#scrollTo=fACSmHLzJanZ
#
# Each thread loops through one chunk of the input array and combines the power consumption
# into one cell of the output array.
# numba.cuda.local.array(shape, type) - create a local array
# numba.cuda.shared.array(shape, type) - create a shared memory array

@cuda.jit
def consolidatePowerConsumption(x, out):
  idx = cuda.grid(1)         # This tells to start with the leftmost index within the block
  stride = cuda.gridsize(1)  # The stride is the same as the width of a block (gridsize)
  for i in range(stride):
    out[idx] = out[idx] + x[i]


# ---------------- CREATE INPUT AND OUTPUT ARRAYS TO PASS TO THE KERNEL ----------------------

A = np.arange(TOTAL_HOUSES,dtype=np.float32)         # Creates an array of floats
B = cuda.pinned_array(TOTAL_HOUSES,dtype=np.float32) # Creates a pinned memory array of floats

# Allocate space on the device for the input data:
d_A = cuda.to_device(A)

# Allocate space on the device for the output data:
d_Out = cuda.device_array_like(d_A)

# CUDA configuration
threads_per_block = 128                                   # blocksize
blocks_per_grid = (TOTAL_HOUSES//threads_per_block) + 1   # gridsize - "//" performs integer division
if(blocks_per_grid > MAX_BLOCKS):
  sys.exit("The number of blocks exceeds the maximum")

# ---------------------------------------- TIMER --------------------------------------------
# We can time our kernels using this:
# https://realpython.com/python-timer/ - will time in seconds
# https://docs.python.org/3/library/time.html#time.perf_counter - there is a nanoseconds option too
startTimer = time.perf_counter()

# TODO - LOOP THROUGH FOR EACH HOUR

# ---------------------- CALL THE KERNEL FUNCTION -------------------------------------------- 
consolidatePowerConsumption[blocks_per_grid, threads_per_block](d_A, d_Out)

# TODO - OUTPUT THE PLOT FOR EACH HOUR
# TODO - SAVE THE PLOT

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