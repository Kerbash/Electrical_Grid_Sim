from numba import cuda, float64
import numpy as np

# INPUT
INPUT_SIZE = 8

# OUTPUT
POOL_SIZE = 2
TOTAL_POOL = 4

DEBUG = True

@cuda.jit
def cudaPool(input, output, last_pool):
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
    cuda.syncthreads()
    # save the sum of the two values (only first thread does this)
    if cuda.threadIdx.x == 0:
        if last_pool: # if this is the last pool, save the sum to output
            output[n_block] = shared[0] + shared[1]
        else: # if this is not the last pool, save the sum to input
            input[cuda.blockIdx.x][cuda.blockIdx.y] = shared[0] + shared[1]


# create input array filled with 1
input = np.ones((INPUT_SIZE, INPUT_SIZE), dtype=np.float64)
# create output array
output = np.zeros((int(INPUT_SIZE / TOTAL_POOL), int(INPUT_SIZE / TOTAL_POOL)), dtype=np.float64)

# allocate memory on the device
d_input = cuda.to_device(input)
d_output = cuda.to_device(output, copy=False) # copy=False because its just an empty array

# main loop
i = POOL_SIZE
while i <= TOTAL_POOL:
    print("POOLING")
    # calculate number of blocks
    dimGrid = (INPUT_SIZE // i, INPUT_SIZE // i)
    dimBlock = (POOL_SIZE)

    if i == TOTAL_POOL:
        last_pool = True
    else:
        last_pool = False

    cudaPool[dimGrid, dimBlock](d_input, d_output, last_pool)

    # copy the input back to the host
    d_input.copy_to_host(input)
    print(input)

    i *= POOL_SIZE

# copy the result back to the host
d_output.copy_to_host(output)

# print the result
print(output)