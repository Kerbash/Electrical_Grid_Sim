import math
from datetime import date, datetime
from random import randint

from tqdm import tqdm

import numpy as np
from opensimplex import OpenSimplex


class home:
    def __init__(self, id, occupancy, coefficient, x_coord, y_coord, distribution_transformer):
        self.id = id
        self.occupancy = occupancy
        self.coefficient = coefficient
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.distribution_transformer = distribution_transformer

    def get_consumption(self, time: datetime, noise=0):
        # convert time and date into a int ================================
        # Create a date object for January 1st of the same year
        jan1 = date(time.year, 1, 1)
        # Calculate the number of days between the two dates
        delta = time.date() - jan1
        # Return the total number of days plus 1 (Julian dates start at 1)
        x = delta.days + 1 * 100

        # get the number of minutes since midnight
        y = time.hour * 60 + time.minute
        x += (y / 1440) * 100

        # TODO: USE ACTUAL EQUATION
        return max(((4.95 * math.sin(
            (2 * math.pi / 100) * x - (math.pi / 2)) + 5) * self.occupancy * self.coefficient) * noise, 0)

    def get_id(self):
        return self.id

    def set_distribution_transformer(self, distribution_transformer):
        self.distribution_transformer = distribution_transformer


def perlin(seed=None, map_height=128, map_width=128):
    if seed is None:
        seed = randint(0, 1000000)

    # Define the size of the noise grid
    width = map_width
    height = map_height

    # Create an OpenSimplex object for generating noise
    noise = OpenSimplex(seed=seed)

    # Generate the noise values for each pixel in the grid
    progess_bar = tqdm(total=width * height)
    noise_values = np.zeros((height, width))
    home_list = []
    for y in range(height):
        for x in range(width):
            noise_values[y][x] = noise.noise2(x=x / width, y=y / height) + 1 / 2
            # row a random number between 0.2 and 0.8
            r = np.random.random() * (0.8 - 0.3) + 0.3
            # if the random number is less than the noise value make it a home
            if r < noise_values[y][x]:
                home_list.append(home(x, 1, 1, x, y, None))
            progess_bar.update(1)

    progess_bar.close()
    return home_list


def make_neighboorhood(n_homes, n_neighboors, random_state=0):
    pass
