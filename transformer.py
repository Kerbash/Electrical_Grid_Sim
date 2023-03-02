
class Transformer:
    def __init__(self, id, capacity, x_coord, y_coord):
        self.transformer_id = id
        #capasity will represent maximum load that transformer can handle
        self.capacity = capacity
        self.x_coord = x_coord
        self.y_coord = y_coord
        #list all homes connected to transformer
        self.homes = []

    def get_id(self):
        return self.transformer_id

    def get_capacity(self):
        return self.capacity

    #add home in transformer
    def add_home(self, home):
        self.homes.append(home)

    #remove home from transformer
    def remove_home(self, home):
        self.homes.remove(home)

    # calculate consumption of all homes connected to transformer
    def get_total_consumption(self, time):
        total_consumption = 0
        #iterate over list of home
        for home in self.homes:
            if home.distribution_transformer == self:
                #calling get home consumption method of class home
                total_consumption += home.get_consumption(time)
        return total_consumption

