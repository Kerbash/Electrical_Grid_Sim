# ---------------------------------------------------------------------------
# # cuda_optimization_with numba.py
# Using the GPU to accelerate an electrical power grid visualization
#
# Date:                   03/14/2023
# Authors:                Pragati Dode, Breanna Powell, and William Selke
#
# +++++++++++++++++ DETAILS ABOUT SYSTEM ++++++++++++++
# IDEs:                   Visual Studio Code; PyCharm
# Processor Used:         11th Gen Intel(R) Core(TM) i7-11700 @ 2.50GHz 2.50 GHz
# GPU Used:               NVIDIA GeForce RTX 3060 Ti
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

import pickle
import time

from matplotlib import pyplot as plt
from numba import cuda, float64
import numpy as np

# nv-nsight-cu-cli --target-processes all --export output.ncu-rep python "cuda_optimizations_with numba.py"

# INPUT
INPUT_SIZE = 4096

# OUTPUT
POOL_SIZE = 2
TOTAL_POOL = 64

DEBUG = True
LOOP_DEBUG = True

@cuda.jit
def cudaPool(input, temp, output, last_pool):
    # set up shared thread with double
    shared = cuda.shared.array((2), dtype=float64)

    # first coordinate
    x1 = cuda.blockIdx.y * POOL_SIZE
    y1 = cuda.blockIdx.x * POOL_SIZE + cuda.threadIdx.x
    # second coordinate
    x2 = x1 + 1
    y2 = y1

    # block number
    n_block = cuda.blockIdx.x * cuda.gridDim.y + cuda.blockIdx.y

    # sum of two values
    shared[cuda.threadIdx.x] = input[x1, y1] + input[x2, y2]
    cuda.threadfence()

    # save the sum of the two values (only first thread does this)
    if cuda.threadIdx.x == 0:
        if last_pool: # if this is the last pool, save the sum to output
            output[cuda.blockIdx.x][cuda.blockIdx.y] = shared[0] + shared[1]
        else: # if this is not the last pool, save the sum to input
            temp[cuda.blockIdx.x][cuda.blockIdx.y] = shared[0] + shared[1]


# load the input array
with open("matrix.pickle", "rb") as f:
    input = pickle.load(f)

# input = np.ones((INPUT_SIZE, INPUT_SIZE), dtype=np.float64)
# create output array
output = np.zeros((int(INPUT_SIZE / TOTAL_POOL), int(INPUT_SIZE / TOTAL_POOL)), dtype=np.float64)
temp = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.float64)

# allocate memory on the device
d_output = cuda.to_device(output, copy=False) # copy=False because it's just an empty array
d_temp = cuda.to_device(temp, copy=False) # copy=False
# start timer
total_time = time.perf_counter()
d_input = cuda.to_device(input)

# main loop
i = POOL_SIZE
# start calculating time
start_time = time.perf_counter()


flip = 0
while i <= TOTAL_POOL:
    # calculate number of blocks
    dimGrid = (INPUT_SIZE // i, INPUT_SIZE // i)
    dimBlock = (POOL_SIZE)

    if i == TOTAL_POOL:
        last_pool = True
    else:
        last_pool = False

    cudaPool[dimGrid, dimBlock](d_input, d_temp, d_output, last_pool)

    # swap the input and output arrays
    f = d_temp
    d_temp = d_input
    d_input = f

    # print the output
    if LOOP_DEBUG:
        # copy the result back to the host
        d_input.copy_to_host(input)

        # cut the input array to size total / i
        temp_2 = input[:int(INPUT_SIZE / i), :int(INPUT_SIZE / i)]

        # every 2 iterations, flip horizontally and rotate 270 degrees
        if flip % 2 == 0:
            print("flip")
            temp_2 = np.flip(temp_2, 1)
            temp_2 = np.rot90(temp_2, 1)

        flip += 1

        # create a heat map from the matrix
        plt.imshow(temp_2, cmap='hot', interpolation='nearest', vmin=0, vmax=pow(i,2))

        # save the heat map
        plt.title("Pool size: " + str(i))
        plt.savefig('heat_map_' + str(i) + '.png')
        plt.show()

    i *= POOL_SIZE

# end calculating time
calc_time = time.perf_counter() - start_time

# copy the result back to the host
d_output.copy_to_host(output)

# end total time
total_time = time.perf_counter() - total_time

# print the end time and total time in microseconds
if DEBUG:
    # print time in microseconds
    print("Total time: " + str(total_time * 1000000) + " microseconds")
    print("Calc time: " + str(calc_time * 1000000) + " microseconds")
    print("Copy time: " + str((total_time - calc_time) * 1000000) + " microseconds")

# create a heat map from the matrix
plt.imshow(output, cmap='hot', interpolation='nearest', vmin=0, vmax=pow(TOTAL_POOL,2))
# save the heat map
plt.title("Output 64 x 64")
plt.savefig('output.png')
plt.show()