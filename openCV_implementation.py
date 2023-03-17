import numpy as np
import cv2
from matplotlib import pyplot as plt


def pool_map(array, intial):
    # compute the maximum value of each 64x64 block
    max_pool = np.max(img_reshaped, axis=(1, 3))

    # convert the resulting array back to an image
    pooled_map = cv2.resize(max_pool, (4096, 4096), interpolation=cv2.INTER_NEAREST)

    return pooled_map

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
    pool = pool_map(matrix)

    # create a heat map from the matrix
    plt.imshow(pool, cmap='hot', interpolation='nearest', vmin=0, vmax=7500)
    # save the heat map
    plt.savefig('heat_map_' + str(i) + '.png')
    plt.show()