import numpy as np
import matplotlib.pyplot as plt
import tqdm
from room import Room
from heater import Heater
from heatsolver import HeatSolver
from boundaryconditions import BoundaryConditions
from heatercontroller import HeaterController

# macierze pochodnych 1 i 2 rzędu (potrzebne)
def D1_left(N):
    return np.eye(N, k = 1) - np.eye(N)

def D1_right(N):
    return np.eye(N) - np.eye(N, k = -1)

def D2(N):
    return np.eye(N, k = 1) + np.eye(N, k = -1) - 2 * np.eye(N)

# tworzenie macierzy ewolucji
# dyskretyzacja czasu
ht = 0.1
t = np.arange(0, 100.0, ht)

# dyskretyzacja przestrzeni
Nx, Ny = 30, 30
x, y = np.linspace(-1, 1, Nx), np.linspace(-1, 1, Ny)
# x, y = np.linspace(0, 3, Nx), np.linspace(0, 3, Ny)
hx, hy = x[1] - x[0], y[1] - y[0]
room = Room(Nx, Ny, hx, hy, -1, -1, 1, 1)
X, Y = np.meshgrid(room.x, room.y)
X_flat, Y_flat = X.flatten(), Y.flatten()

# pierwsza przykładowa pozycja grzejnika
# heater_mask = np.where((Y_flat > -0.8) & (Y_flat < -0.6) & (X_flat > -0.3) & (X_flat < 0.3), True, False)
heater_mask_2d = np.where((Y > -0.8) & (Y < -0.6) & (X > -0.3) & (X < 0.3), True, False)
heater_mask = heater_mask_2d.flatten()
P = 915.0  # "moc" grzejnika – skalowanie
p = 101325.0 # ciśnienie powietrza w pascalach
r = 287.05 # J/(kg K)
c = 1005 # ciepło właściwe suchego powietrza
A_heater = 0.06
heater_power = (3600 * P * r )/ (p * A_heater * c)
# heater_power = 200
heater = Heater(heater_mask, heater_power, 298)
room.add_heater("heater 1", heater)
lambda_window = 0.96
window = np.where((Y_flat == -1) & (X_flat > -0.3) & (X_flat < 0.3), True, False)
room.add_window("window bottom 1", window, lambda_window)

# heater = ((Y_flat < 0.05) & (Y_flat > 0.15) & (X_flat > 1.5) & (X_flat < 2.5))
# window = ((Y_flat == y[0]) & (X_flat > 1) & (X_flat < 3))


# identyfikacja punktów brzegowych w spłaszczonych wektorach przestrzeni
lambda_mat = 0.6   # przewodność ściany
ind_brzeg_left = np.where(X_flat == x[0], True, False)
room.add_walls("left 1", ind_brzeg_left, lambda_mat)
ind_brzeg_right = np.where(X_flat == x[-1], True, False)
room.add_walls("right 1", ind_brzeg_right, lambda_mat)
ind_brzeg_bottom_wall = np.where((Y_flat == y[0]) & ((X_flat < -0.3) | (X_flat > 0.3)), True, False)
room.add_walls("bottom 1", ind_brzeg_bottom_wall, lambda_mat)
# ind_brzeg_bottom_wall = np.where((Y_flat == y[0]) & ((X_flat < 1) | (X_flat > 3)), True, False)
ind_brzeg_top = np.where(Y_flat == y[-1], True, False)
room.add_walls("top 1", ind_brzeg_top, lambda_mat)

solver = HeatSolver(room, alpha = 0.0684, ht = 0.1)
bc = BoundaryConditions(room)
controller = HeaterController(room)

solver.apply_boundary_conditions(bc)

u0 = np.ones(len(X_flat)) * 292
u = u0.copy()

indices = ind_brzeg_left | ind_brzeg_right | ind_brzeg_top | ind_brzeg_bottom_wall

for time in tqdm.tqdm(t):
    source = controller.compute_source(u)
    u = solver.step(u, source, bc)

levels = np.linspace(u.min(), u.max(), 50)

plt.contourf(X, Y, u.reshape(Nx, Ny), levels = levels)
plt.title("Wynik końcowy")
plt.colorbar()

plt.show()

print(max(u))
print(u.mean())
