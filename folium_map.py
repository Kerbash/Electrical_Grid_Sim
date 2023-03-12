"""
Using the built in python library pyTorch, two different simulation functions are performed.
One with and without the use of PinnedMemory. The pre-calculated electrical usage are loaded from pickle
files. From there the function is then called to pool the matrix into a smaller matrix for rendering.

Two different version of the pyTorch function is used. The first is the Naive version in which the
function max_pool2d is used. The second is, also used the max_pool2d function, but with an added pinned
memory before initializing the GPU. The pinned memory is used to transfer the data from the CPU to the
GPU. This is done to avoid the overhead of the CPU to GPU transfer.

The results are then retrived. Using the .cpu() function to transfer the data from the GPU to the CPU.
"""