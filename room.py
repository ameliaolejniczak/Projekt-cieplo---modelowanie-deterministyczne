import numpy as np

class Room:
    def __init__(self, nx, ny, hx, hy, min_x, min_y, max_x, max_y):
        self.nx = nx
        self.ny = ny
        self.hx = hx
        self.hy = hy
        self.x, self.y = np.linspace(min_x, max_x, nx), np.linspace(min_y, max_y, ny)

        self.heaters = {}
        self.walls = {}
        self.windows = {}

    def add_heater(self, name, heater):
        self.heaters[name] = heater

    def add_walls(self, name, mask, material_params):
        self.walls[name] = (mask, material_params)

    def add_window(self, name, mask, material_params):
        self.windows[name] = (mask, material_params)





