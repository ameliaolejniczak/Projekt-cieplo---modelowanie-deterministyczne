class Heater:
    def __init__(self, mask, power, setpoint):
        self.mask = mask
        self.power = power
        self.setpoint = setpoint

    def is_on(self, avg_temp):
        return avg_temp < self.setpoint

