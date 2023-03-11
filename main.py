#---------------------------------------------------------------------------
# MAIN.PY
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
# $ conda install cudatoolkit
# $ conda install tqdm
#
# Use the following commands if using pip:
# $ pip install cuda-python
#
# +++++++++++++++++ LIBRARY USED +++++++++++++++++ 
# Numba library information: https://numba.readthedocs.io/en/stable/cuda/overview.html
# Numba library contents: https://numba.readthedocs.io/en/stable/cuda/index.html
# Note: Numba does not implement: dynamic parallelism and texture memory

from datetime import datetime
from random import random
from random import random
from tqdm import tqdm

from home import home, perlin
import matplotlib.pyplot as plt

# MAP SIZE x MAP SIZE
MAP_SIZE = 512

# Perlin is for noise: https://pypi.org/project/perlin/
homes = perlin(seed=200, map_width=MAP_SIZE, map_height=MAP_SIZE)

# tqdm is a progress bar: https://www.educative.io/answers/what-is-tqdm-library-in-python
tqdm = tqdm(total=24*len(homes)) 

# -------- UPDATE THE PLOT FOR EVERY HOUR TO CREATE THE SIMULATION -------------
for hour in range(24):
    datetime_now = datetime(2020, 1, 1, hour, 0, 0)
    # plot the home in the graph
    total_consumption = 0
    # clear the plot
    plt.clf() # clf = clear the figure: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.clf.html
    
    # ------------------- INCREMENT THE TOTAL CONSUMPTION --------------------
    for home in homes:
        # plot with square as marker
        # create noise 0.9 - 1.1
        noise = 0.9 + (1.1 - 0.9) * random()
        # set the color based on the current consumption
        cur = home.get_consumption(datetime_now, noise)

        # ---- TAKE THE LARGE DATASET, COMBINE MULTIPLE HOUSES, NAIVE VERSION ----
        # TO DO: CREATE A LARGE NUMBER OF HOUSES
        # TO DO: IMPLEMENT A NAIVE STENCIL METHOD THAT COMBINES SEVERAL HOUSES SO THAT 
        #        A LARGE NUMBER OF HOUSES IS COMBINED AND FITS WITHIN A SMALL HEATMAP.

        total_consumption += cur
        # set closer to red if the consumption is higher green if low
        color = (1 - min(cur / 10, 1), 1 - min(cur / 10, 1), 1)
        plt.plot(home.x_coord, home.y_coord, marker='s', color=color)
        # set plot to Map Size
        tqdm.update(1)

    plt.axis([0, MAP_SIZE, 0, MAP_SIZE])

    # add a small total_consumption text to the plot
    plt.text(10, MAP_SIZE-10, 'Total Consumption: ' + str(total_consumption))
    # set title as time
    plt.title(datetime_now)
    # save plot to file
    plt.show()
    # plt.savefig(str(hour) + '.png')

tqdm.close()
