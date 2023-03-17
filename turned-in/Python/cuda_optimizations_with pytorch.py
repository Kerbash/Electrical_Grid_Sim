# ---------------------------------------------------------------------------
# cuda_optimization_with pytorch.py
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

from random import random

# Needs both CUDA
# Needs to run CUDA on version 11.7 and torch 1.13.1
# pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117

# nv-nsight-cu-cli --target-processes all --export output.ncu-rep python "cuda_optimizations_with pytorch.py"


MATRIX_SIZE = 32768



KERNEL_SIZE = 512





import torch
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt

DEBUG = True


def pytorch_builtin_pooling(input_tensor, pool_size):
    """
    Performs max pooling on a 2D input tensor using a square pooling window with equal stride and kernel size.
    This function uses CUDA optimizations for improved performance, including pinned memory and the PyTorch
    `torch.nn.functional.max_pool2d` function.

    Args:
        - input_tensor: a 2D PyTorch tensor
        - pool_size: an integer specifying the size of the square pooling window

    Returns:
        - a 2D PyTorch tensor of pooled values
    """
    # Check that the pool size is valid
    if pool_size < 1:
        raise ValueError("Pool size must be at least 1.")

    # Compute the output dimensions
    input_height, input_width = input_tensor.size()
    output_height = (input_height - pool_size) // pool_size + 1
    output_width = (input_width - pool_size) // pool_size + 1

    # start main timer
    total_time = time.perf_counter()
    # Create a tensor to hold the output and move it to the GPU
    output_gpu = torch.empty((output_height, output_width), dtype=torch.double, device='cuda')

    input_gpu = input_tensor.to(device='cuda')

    calc_time = time.perf_counter()
    # Perform max pooling on the GPU
    output_gpu = F.max_pool2d(input_gpu.view(1, 1, input_height, input_width), pool_size).view(output_height, output_width)
    calc_time = time.perf_counter() - calc_time

    # copy output to CPU
    output_cpu = output_gpu.to(device='cpu')
    # end main timer
    total_time = time.perf_counter() - total_time

    if DEBUG:
        # print time in microseconds
        print("Total time: " + str(total_time * 1000000) + " microseconds")
        print("Calc time: " + str(calc_time * 1000000) + " microseconds")
        print("Copy time: " + str((total_time - calc_time) * 1000000) + " microseconds")

    return output_cpu


def pytorch_custom_pooling(input_tensor, pool_size):
    """
    Performs max pooling on a 2D input tensor using a square pooling window with equal stride and kernel size.
    :param input_tensor:
    :param pool_size:
    :return:
    """
    # Check that the pool size is valid
    if pool_size < 1:
        raise ValueError("Pool size must be at least 1.")

    # Compute the output dimensions
    input_height, input_width = input_tensor.size()
    output_height = (input_height - pool_size) // pool_size + 1
    output_width = (input_width - pool_size) // pool_size + 1

    # Create a tensor to hold the output
    input_pinned = torch.empty((input_height, input_width), dtype=torch.float32, pin_memory=True)
    output_pinned = torch.empty((output_height, output_width), dtype=torch.float32, pin_memory=True)

    # start main timer
    total_time = time.perf_counter()

    # Copy input tensor to pinned memory on the CPU
    input_pinned.copy_(input_tensor)

    # Transfer pinned memory tensors to the GPU
    input_gpu = input_pinned.to(device='cuda', non_blocking=True)

    # Perform max pooling on the GPU
    calc_time = time.perf_counter()
    output_gpu = F.max_pool2d(input_gpu.view(1, 1, input_height, input_width), pool_size).view(output_height, output_width)
    calc_time = time.perf_counter() - calc_time

    # Transfer output tensor back to pinned memory on the CPU
    output_pinned.copy_(output_gpu.cpu())

    # end main timer
    total_time = time.perf_counter() - total_time

    if DEBUG:
        # print time in microseconds
        print("Total time: " + str(total_time * 1000000) + " microseconds")
        print("Calc time: " + str(calc_time * 1000000) + " microseconds")
        print("Copy time: " + str((total_time - calc_time) * 1000000) + " microseconds")

    return output_pinned


for i in range(1):
    """    # Create a 2D tensor of random values
    input_tensor = torch.rand(MATRIX_SIZE, MATRIX_SIZE, dtype=torch.double)
    """
    # load map from unpickle
    import pickle

    # generate a matrix of size MATRIX_SIZE x MATRIX_SIZE filled with random float
    matrix = [[random() for x in range(MATRIX_SIZE)] for y in range(MATRIX_SIZE)]


    # convert to torch tensor
    input_tensor = torch.tensor(matrix, dtype=torch.double)
    pool = pytorch_custom_pooling(input_tensor, KERNEL_SIZE)

    # create a heat map from the matrix
    plt.imshow(pool, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    # save the heat map
    plt.savefig('heat_map_' + str(i) + '.png')
    plt.show()
