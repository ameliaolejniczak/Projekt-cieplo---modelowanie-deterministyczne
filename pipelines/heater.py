class Heater:
    '''Klasa tworzy grzejnik'''
    def __init__(self, mask, power, setpoint):
        self.mask = mask
        self.power = power
        self.setpoint = setpoint

    def is_on(self, avg_temp):
        '''Funkcja zwraca True lub False w zaleznosci od tego, czy grzejnik
        powinien byc wlaczony na podstawie sredniej temperatury w pokoju'''
        return avg_temp < self.setpoint

