import math
from datetime import date, datetime


class home:
    def __init__(self, id, occupancy, coefficient, x_coord, y_coord, distribution_transformer):
        self.id = id
        self.occupancy = occupancy
        self.coefficient = coefficient
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.distribution_transformer = distribution_transformer

    def get_consumption(self, time : datetime, noise = 0):
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
        return max(((4.95 * math.sin((2 * math.pi / 100) * x - (math.pi / 2)) + 5) * self.occupancy * self.coefficient) * noise, 0)

    def get_id(self):
        return self.id

    def set_distribution_transformer(self, distribution_transformer):
        self.distribution_transformer = distribution_transformer
