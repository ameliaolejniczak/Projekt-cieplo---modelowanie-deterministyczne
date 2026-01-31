import numpy as np

class HeaterController:
    def __init__(self, room):
        self.room = room

    def compute_source(self, u):
        source = np.zeros_like(u)
        for name, heater in self.room.heaters.items():
            avg_temp = np.mean(u[heater.mask])
            if heater.is_on(avg_temp):
                source[heater.mask] = heater.power

        return source
