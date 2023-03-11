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
TOTAL_HOUSES = 320000                           
BLOCK_SIZE = 128                          # NUMBER OF THREADS PER BLOCK
GRID_SIZE = TOTAL_HOUSES//BLOCK_SIZE      # NUMBER OF BLOCKS PER GRID = 320000 / 128 THREADS = 2,500 BLOCKS
MAP_SIZE = math.sqrt(GRID_SIZE)           # 50 BY 50 SQUARE HEATMAP (sqrt(BLOCKS)

# Hardware Constraints:
MAX_BLOCKS = 65535

# CUDA configuration - grid size
if(GRID_SIZE > MAX_BLOCKS):
  sys.exit("The number of blocks exceeds the maximum")

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

#In = np.empty(TOTAL_HOUSES,dtype=np.float64)        # Creates an input array of 64-bit precision
In = cuda.pinned_array(TOTAL_HOUSES,dtype=np.float64)  # Creates a pinned memory array of 64-bit precision

# Randon numbers: https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html
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

Out = np.empty(GRID_SIZE, dtype=np.float64) # Creates an output array of 64-bit precision

# -------- UPDATE THE PLOT FOR EVERY HOUR TO CREATE THE SIMULATION -------------
for hour in range(24):

  # Allocate space on the device for the output data:
  d_Out = cuda.device_array_like(Out)
    
  # ---------------------- CALL THE KERNEL FUNCTION -------------------------------------------- 
  consolidatePowerConsumption[GRID_SIZE, BLOCK_SIZE](d_In, d_Out)

  # Wait for all threads to complete
  # cuda.synchronize()

  # Out = np.empty(shape=d_Out.shape, dtype=d_Out.dtype)
  d_Out.copy_to_host(Out) # Copy contents from the device d_Out to host Out
  # Out is an ndarray https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html

  # Clear the plot
  plt.clf() # clf = clear the figure: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.clf.html
  total_consumption = 0 # Total power consumption (for the plot) - reset with each loop
  
  for blockId in range(Out.size):
    cur = Out[int(blockId)]       # The power consumption for that one block in that one hour
    total_consumption += cur      # The power consumption total

    xcoord = blockId // MAP_SIZE  # ex) if the blockId is 1501, then 1501 / 50 = x coordinate of 30
    ycoord = blockId % MAP_SIZE   # ex) if the blockId is 1501, then 1501 % 50 = y coordinate of 1
    
    # set closer to red if the consumption is higher green if low
    color = (1 - min(cur / 10, 1), 1 - min(cur / 10, 1), 1)
    plt.plot(xcoord, ycoord, marker='s', color=color)
    
  # set plot to Map Size
  plt.axis([0, MAP_SIZE, 0, MAP_SIZE])
  
  # add a small total_consumption text to the plot
  plt.text(10, MAP_SIZE-10, 'Total Consumption: ' + str(total_consumption))
  plt.text(10, MAP_SIZE-20, 'Hour: ' + str(hour))

  # set title
  plt.title("Power Consumption in a City of 360,000 Houses")

  # plt.show()
  
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