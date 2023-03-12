

# Needs both CUDA
# Needs to run CUDA on version 11.7 and torch 1.13.1
# pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117

# nv-nsight-cu-cli --target-processes all --export output.ncu-rep python "cuda_optimizations_with pytorch.py"

MATRIX_SIZE = 4096
KERNEL_SIZE = 64

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

    # Perform max pooling on the GPU
    input_gpu = input_tensor.to(device='cuda')

    calc_time = time.perf_counter()
    output_gpu = torch.nn.functional.max_pool2d(input_gpu.view(1, 1, input_height, input_width), pool_size).view(
        output_height, output_width)
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
    output_gpu = torch.nn.functional.max_pool2d(input_gpu.view(1, 1, input_height, input_width), pool_size).view(
        output_height, output_width)
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

    with open('matrix.pickle', 'rb') as handle:
        matrix = pickle.load(handle)
    # multiply the matrix by i
    matrix = matrix

    # convert to torch tensor
    input_tensor = torch.tensor(matrix, dtype=torch.double)
    pool = pytorch_builtin_pooling(input_tensor, KERNEL_SIZE)

    # create a heat map from the matrix
    plt.imshow(pool, cmap='hot', interpolation='nearest', vmin=0, vmax=7500)
    # save the heat map
    plt.savefig('heat_map_' + str(i) + '.png')
    plt.show()
