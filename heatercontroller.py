import numpy as np

class HeaterController:
    def __init__(self, room):
        self.room = room

    def compute_source(self, u):
        source = np.zeros_like(u)

        for name, heater in self.room.heaters.items():
            avg_temp = np.mean(u[heater.mask])
            rho = 101325.0 / (287.05 * np.mean(u[heater.mask]))
            c = 1005.0
            V_cell = self.room.hx * self.room.hy * 2.5
            n = np.sum(heater.mask)
            if heater.is_on(avg_temp):
                S = heater.power / (rho * c * V_cell * n)
                source[heater.mask] = S

        return source
