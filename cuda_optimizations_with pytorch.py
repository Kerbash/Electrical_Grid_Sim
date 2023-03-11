import torch
import numpy as np

# Needs both CUDA and CuDNN

MATRIX_SIZE = 1024
KERNEL_SIZE = 32

import torch
import torch.nn.functional as F

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

    # Perform max pooling using the torch.nn.functional.max_pool2d function
    output = F.max_pool2d(input_tensor.view(1, 1, input_height, input_width), pool_size).view(output_height,
                                                                                              output_width)
    return output

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
    input_pinned = torch.empty((input_height, input_width), dtype=torch.double, pin_memory=True)
    output_pinned = torch.empty((output_height, output_width), dtype=torch.double, pin_memory=True)

    # Copy input tensor to pinned memory on the CPU
    input_pinned.copy_(input_tensor)

    # Transfer pinned memory tensors to the GPU
    input_gpu = input_pinned.to(device='cuda', non_blocking=True)
    output_gpu = output_pinned.to(device='cuda', non_blocking=True)

    # Perform max pooling on the GPU
    output_gpu = torch.nn.functional.max_pool2d(input_gpu.view(1, 1, input_height, input_width), pool_size).view(
        output_height, output_width)

    # Transfer output tensor back to pinned memory on the CPU
    output_pinned.copy_(output_gpu.cpu())

    return output_pinned


# Create a 2D tensor of random values
input_tensor = torch.rand(MATRIX_SIZE, MATRIX_SIZE)
pool = pytorch_custom_pooling(input_tensor, KERNEL_SIZE)