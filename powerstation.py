class powerstation:
    def __init__(self, id, transformer_count, x_coord, y_coord):
        self.id = id
        self.transformer_count = transformer_count
        self.transformer_list = [transformer_count]
        self.power_generated_per_hour = []
        self.x_coord = x_coord
        self.y_coord = y_coord

    # -------------------- getters -----------------------#
    def get_id(self):
        return self.id
    
    def get_transformer_count(self):
        return self.transformer_count

    def get_x_coord(self):
        return self.get_x_coord

    def get_y_coord(self):
        return self.get_y_coord

    # -------------------- setters -----------------------#
    def set_transformer_count(self, transformer_count):
        self.transformer_count = transformer_count

    def set_transformer_list(self, transformer_list):
        # https://www.geeksforgeeks.org/copy-python-deep-copy-shallow-copy/
        self.transformer_list = transformer_list.deepcopy()

    def power_generated_per_hour(self, power_generated_per_hour):
        self.power_generated_per_hour = power_generated_per_hour.deepcopy()

    