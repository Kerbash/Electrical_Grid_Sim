import torch

# create a matrix with random values
matrix = torch.randn(1024, 1024, dtype=torch.float32).cuda(non_blocking=True)


# torch.empty_like creates new tensor of same size and data type like input tensor.
#pin_memory=True This flag is used to allcate output in pinned memory, which can help optimize data transfer between GPU and CPU
#pin_memory=True this flag is used to specify GPU to CPU transfer should be non blocking.
# allocate pinned memory for the output
output = torch.empty_like(matrix, pin_memory=True).cuda(non_blocking=True)

# loop over the rows of the matrix.
#matrix.shape[0] = row and matrix.shape[1]=col
for i in range(matrix.shape[0]):
    # copy the row to pinned memory
    row = matrix[i].pin_memory()

# In case want to loop over the columns of the heat map
#     for j in range(matrix.shape[1]):
#         # copy the column to pinned memory
#         col = matrix[:, j].pin_memory()

    # perform operation on the row
    row_mean = torch.mean(row)

    # copy the result back to regular memory
    output[i] = row_mean

    # unpin the row
    row.unpin_memory()

# free pinned memory.Not strictly necessary, but recommended
output.pin_memory()
del output
