# INSTALLATION INSTRUCTIONS:
# https://numba.readthedocs.io/en/stable/cuda/overview.html

# Use the following command if using Conda:
# $ conda install cudatoolkit

# Info about the numba library: https://numba.readthedocs.io/en/stable/cuda/index.html

from numba import cuda

@cuda.jit
def kernel(x, out):
  idx = cuda.grid(1)
  # Example of a square root function from https://thedatafrog.com/en/articles/cuda-kernel-python/
  # out[idx] = math.sqrt(x[idx])

