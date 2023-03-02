from datetime import datetime
from random import random
from random import random
from tqdm import tqdm

from home import home, perlin
import matplotlib.pyplot as plt

# MAP SIZE x MAP SIZE
MAP_SIZE = 512

homes = perlin(seed=200, map_width=MAP_SIZE, map_height=MAP_SIZE)


tqdm = tqdm(total=24*len(homes))
for hour in range(24):
    datetime_now = datetime(2020, 1, 1, hour, 0, 0)
    # plot the home in the graph
    total_consumption = 0
    plt.clf()
    for home in homes:
        # plot with square as marker
        # create noise 0.9 - 1.1
        noise = 0.9 + (1.1 - 0.9) * random()
        # set the color based on the current consumption
        cur = home.get_consumption(datetime_now, noise)
        total_consumption += cur
        # set closer to red if the consumption is higher green if low
        color = (1 - min(cur / 10, 1), 1 - min(cur / 10, 1), 1)
        plt.plot(home.x_coord, home.y_coord, marker='s', color=color)
        # set plot to Map Size
        tqdm.update(1)

    plt.axis([0, MAP_SIZE, 0, MAP_SIZE])
    # clear the plot
    # add a small total_consumption text to the plot
    plt.text(10, MAP_SIZE-10, 'Total Consumption: ' + str(total_consumption))
    # set title as time
    plt.title(datetime_now)
    # save plot to file
    plt.savefig(str(hour) + '.png')

tqdm.close()

