import numpy as np

class HeaterController:
    '''Klasa przechowujaca informacje o zrodle ciepla'''
    def __init__(self, room):
        self.room = room

    def compute_source(self, u):
        '''Funkcja tworzaca zrodlo ciepla i przeliczajaca jego moc'''
        source = np.zeros_like(u)

        for name, heater in self.room.heaters.items():
            avg_temp = np.mean(u)
            temp_to_rho = np.min([np.mean(u[heater.mask]), 348])
            rho = 101325.0 / (287.05 * temp_to_rho)
            c = 1005.0
            V_cell = self.room.hx * self.room.hy
            n = np.sum(heater.mask)
            if heater.is_on(avg_temp) and np.mean(u[heater.mask]) < 348:
                S = heater.power / (rho * c * V_cell * n)
                source[heater.mask] = S

        return source
