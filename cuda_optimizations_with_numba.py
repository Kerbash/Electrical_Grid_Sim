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
import matplotlib.pyplot as plt

# Sizes:
CITY_WIDTH = 4096           # 4096 * 4096 = 16,777,216 total houses          
                            # Note: There does not need to be a thread for every house.
                            # There just needs to be a thread for each heatmap cell.
                                           
BLOCK_SIZE = 64                                # NUMBER OF THREADS PER BLOCK
MAP_SIZE = 64                                  # SIZE OF SQUARE HEATMAP = 64 * 64 = 4096 heatmap cells to fill
GRID_SIZE = (MAP_SIZE * MAP_SIZE)//BLOCK_SIZE  # NUMBER OF BLOCKS PER GRID = 4096 / 64 THREADS = 64 BLOCKS

# Learn more about our GPU using this function:
# cuda.detect()

# --------------------------------- KERNEL FUNCTION --------------------------------------
# See example kernel functions: 
# https://thedatafrog.com/en/articles/cuda-kernel-python/
# https://colab.research.google.com/github/cbernet/maldives/blob/master/numba/numba_cuda_kernel.ipynb#scrollTo=fACSmHLzJanZ
# https://numba.readthedocs.io/en/stable/cuda/examples.html#matrix-multiplication

# Each thread loops through one chunk of the input array and combines the power consumption
# into one cell of the output array.
# numba.cuda.local.array(shape, type) - create a local array
# numba.cuda.shared.array(shape, type) - create a shared memory array

@cuda.jit
def consolidatePowerConsumption(In, Out):
  tile = cuda.gridsize(1)  # The tile stride is the same as the width of a block (gridsize)
  print(tile)

  sharedIn = cuda.shared.array(shape=(tile, tile), dtype=np.float64) # Each thread takes care of 1 tile of the input matrix.
  localOut = np.float64(0.) # Each thread takes care of 1 cell of the heatmap.
  
  x, y = cuda.grid(2)
  idx = cuda.threadIdx.x
  idy = cuda.threadIdy.y
  
  for i in range(tile):

    # Use the shared memory
    sharedIn[idy, idx] = 0
    if y < In.shape[0] and (idx + i * BLOCK_SIZE) < In.shape[1]:
      sharedIn[idy, idx] = In[y, idx + i * BLOCK_SIZE] # Load the data to shared.

    # Wait for load to finish
    cuda.syncthreads()

    for j in range(BLOCK_SIZE):
      localOut += sharedIn[idy, j]

    # Wait until all threads finish computing
    cuda.syncthreads()
  
  if y < Out.shape[0] and x < Out.shape[1]:
    Out[y, x] = localOut

# ---------------- CREATE INPUT AND OUTPUT ARRAYS TO PASS TO THE KERNEL ----------------------
# max_pinned_size = cuda.max_pinned_bytes()
# if (CITY_WIDTH*CITY_WIDTH*np.float64) > max_pinned_size:
#   sys.exit("The city size exceeds pinned memory size")

cityShape = (CITY_WIDTH, CITY_WIDTH)
#In = np.empty(TOTAL_HOUSES, dtype=np.float64)       # Creates an input array of 64-bit precision
In = cuda.pinned_array(cityShape, dtype=np.float64)  # Creates a pinned memory array of 64-bit precision

# Random numbers: https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html
mean = 12      # mean
stdDev = 5.5   # standard deviation for random numbers
In[:] = mean + stdDev * np.random.randn(*In.shape)


# ---------------------------------------- TIMER --------------------------------------------
# We can time our kernels using this:
# https://realpython.com/python-timer/ - will time in seconds
# https://docs.python.org/3/library/time.html#time.perf_counter - there is a nanoseconds option too
startTimer = time.perf_counter()

# Allocate space on the device for the input data:
d_In = cuda.to_device(In)

neighborhoodShape = (GRID_SIZE, GRID_SIZE)
Out = np.empty(neighborhoodShape, dtype=np.float64) # Creates an output array of 64-bit precision

# Allocate space on the device for the output data:
d_Out = cuda.device_array_like(Out)

# -------- UPDATE THE PLOT FOR EVERY HOUR TO CREATE THE SIMULATION -------------
# TODO: set to range(24) for 24 hours
for hour in range(1):

  d_Out[:] = 0 # Clear the output array for each hour

  # ---------------------- CALL THE KERNEL FUNCTION -------------------------------------------- 
  consolidatePowerConsumption[GRID_SIZE, BLOCK_SIZE](d_In, d_Out)

  # Out = np.empty(shape=d_Out.shape, dtype=d_Out.dtype)
  d_Out.copy_to_host(Out) # Copy contents from the device d_Out to host Out
  # Out is an ndarray https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html

  # Clear the plot
  plt.clf() # clf = clear the figure: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.clf.html
  #total_consumption = 0 # Total power consumption (for the plot) - reset with each loop
  
  # create a heat map from the matrix
  plt.imshow(Out, cmap='hot', interpolation='nearest')
  
  # add a small total_consumption text to the plot
  # plt.text(10, MAP_SIZE-10, 'Total Consumption: ' + str(total_consumption))
  plt.text(10, MAP_SIZE-20, 'Hour: ' + str(hour))

  # set title
  plt.title("Power Consumption in a City of 360,000 Houses")

  plt.show()
  
stopCompTime = time.perf_counter()

# save plot to file
# plt.savefig('simulation_with_numba.png')

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