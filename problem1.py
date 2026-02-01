import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tqdm
from room import Room
from heater import Heater
from heatsolver import HeatSolver
from boundaryconditions import BoundaryConditions
from heatercontroller import HeaterController

# tworzenie macierzy ewolucji
# dyskretyzacja czasu
ht = 1
t = np.arange(0, 7200.0, ht)

# dyskretyzacja przestrzeni
Nx, Ny = 30, 30
x, y = np.linspace(0, 3, Nx), np.linspace(0, 3, Ny)
# x, y = np.linspace(0, 3, Nx), np.linspace(0, 3, Ny)
hx, hy = x[1] - x[0], y[1] - y[0]
room = Room(Nx, Ny, hx, hy, 0, 0, 3, 3)
X, Y = np.meshgrid(room.x, room.y)
X_flat, Y_flat = X.flatten(), Y.flatten()

# pierwsza przykładowa pozycja grzejnika
# heater_mask = np.where((Y_flat > -0.8) & (Y_flat < -0.6) & (X_flat > -0.3) & (X_flat < 0.3), True, False)
# heater_mask = np.where((Y_flat > 0.05) & (Y_flat < 0.15) & (X_flat > 1) & (X_flat < 2), True, False)
heater_mask = np.where((Y_flat < 2.9) & (Y_flat > 2.8) & (X_flat > 1) & (X_flat < 2), True, False)
P = 1267.0
heater_power = P
# heater_power = 200
heater = Heater(heater_mask, heater_power, 298)
room.add_heater("heater 1", heater)
lambda_window = 0.96
#window = np.where((Y_flat == 1) & (X_flat > -0.3) & (X_flat < 0.3), True, False)
window = np.where((Y_flat == y[-1]) & (X_flat > 1) & (X_flat < 2), True, False)
room.add_window("window top 1", window, lambda_window)



# identyfikacja punktów brzegowych w spłaszczonych wektorach przestrzeni
lambda_mat = 1.7   # przewodność ściany
ind_brzeg_left = np.where(X_flat == x[0], True, False)
room.add_walls("left 1", ind_brzeg_left, lambda_mat)
ind_brzeg_right = np.where(X_flat == x[-1], True, False)
room.add_walls("right 1", ind_brzeg_right, lambda_mat)
ind_brzeg_bottom = np.where((Y_flat == y[0]), True, False)
room.add_walls("bottom 1", ind_brzeg_bottom, lambda_mat)
ind_brzeg_top = np.where((Y_flat == y[-1]) & ((X_flat < 1) | (X_flat > 2)), True, False)
# ind_brzeg_top = np.where(((Y_flat == y[-1]) & ((X_flat < -0.3) | (X_flat > 0.3))), True, False)
room.add_walls("top 1", ind_brzeg_top, lambda_mat)

solver = HeatSolver(room, alpha = 19.0 * 10**(-5), ht = 0.1)
bc = BoundaryConditions(room)
controller = HeaterController(room)

solver.apply_boundary_conditions(bc)

u0 = np.ones(len(X_flat)) * 292
u = u0.copy()

indices = ind_brzeg_left | ind_brzeg_right | ind_brzeg_top | ind_brzeg_bottom

frames = []

for i, time in enumerate(tqdm.tqdm(t)):
    source = controller.compute_source(u)
    u = solver.step(u, source, bc)

    if i % 10 == 0:
        frames.append(u.copy())


levels = np.linspace(u.min(), u.max(), 50)

plt.contourf(X, Y, u.reshape(Nx, Ny), levels = levels)
plt.title("Wynik końcowy")
plt.colorbar()

plt.show()

print(max(u - 273))
print((u - 273).mean())

fig, ax = plt.subplots()
levels = np.linspace(u.min(), u.max(), 50)
contour = ax.contourf(X, Y, frames[0].reshape(Nx, Ny), levels=levels)

def update(frame):
    ax.clear()
    contour = ax.contourf(X, Y, frame.reshape(Nx, Ny), levels=levels)
    return contour

anim = FuncAnimation(fig, update, frames=frames, interval=50)
plt.show()

ht = 1
t = np.arange(0, 7200.0, ht)

# dyskretyzacja przestrzeni
Nx, Ny = 30, 30
x, y = np.linspace(0, 3, Nx), np.linspace(0, 3, Ny)
# x, y = np.linspace(0, 3, Nx), np.linspace(0, 3, Ny)
hx, hy = x[1] - x[0], y[1] - y[0]
room = Room(Nx, Ny, hx, hy, 0, 0, 3, 3)
X, Y = np.meshgrid(room.x, room.y)
X_flat, Y_flat = X.flatten(), Y.flatten()

# pierwsza przykładowa pozycja grzejnika
# heater_mask = np.where((Y_flat > -0.8) & (Y_flat < -0.6) & (X_flat > -0.3) & (X_flat < 0.3), True, False)
# heater_mask = np.where((Y_flat > 0.05) & (Y_flat < 0.15) & (X_flat > 1) & (X_flat < 2), True, False)
heater_mask = np.where((Y_flat > 0.1) & (Y_flat < 0.2) & (X_flat > 1) & (X_flat < 2), True, False)
P = 1267.0
heater_power = P
# heater_power = 200
heater = Heater(heater_mask, heater_power, 298)
room.add_heater("heater 1", heater)
lambda_window = 0.96
#window = np.where((Y_flat == 1) & (X_flat > -0.3) & (X_flat < 0.3), True, False)
window = np.where((Y_flat == y[-1]) & (X_flat > 1) & (X_flat < 2), True, False)
room.add_window("window top 1", window, lambda_window)



# identyfikacja punktów brzegowych w spłaszczonych wektorach przestrzeni
lambda_mat = 1.7   # przewodność ściany
ind_brzeg_left = np.where(X_flat == x[0], True, False)
room.add_walls("left 1", ind_brzeg_left, lambda_mat)
ind_brzeg_right = np.where(X_flat == x[-1], True, False)
room.add_walls("right 1", ind_brzeg_right, lambda_mat)
ind_brzeg_bottom = np.where((Y_flat == y[0]), True, False)
room.add_walls("bottom 1", ind_brzeg_bottom, lambda_mat)
ind_brzeg_top = np.where((Y_flat == y[-1]) & ((X_flat < 1) | (X_flat > 2)), True, False)
# ind_brzeg_top = np.where(((Y_flat == y[-1]) & ((X_flat < -0.3) | (X_flat > 0.3))), True, False)
room.add_walls("top 1", ind_brzeg_top, lambda_mat)

solver = HeatSolver(room, alpha = 19.0 * 10**(-5), ht = ht)
bc = BoundaryConditions(room)
controller = HeaterController(room)

solver.apply_boundary_conditions(bc)

u0 = np.ones(len(X_flat)) * 292
u = u0.copy()

indices = ind_brzeg_left | ind_brzeg_right | ind_brzeg_top | ind_brzeg_bottom

frames = []

for i, time in enumerate(tqdm.tqdm(t)):
    source = controller.compute_source(u)
    u = solver.step(u, source, bc)

    if i % 10 == 0:
        frames.append(u.copy())


levels = np.linspace(u.min(), u.max(), 50)

plt.contourf(X, Y, u.reshape(Nx, Ny), levels = levels)
plt.title("Wynik końcowy")
plt.colorbar()

plt.show()

print(max(u - 273))
print((u - 273).mean())

fig, ax = plt.subplots()
levels = np.linspace(u.min(), u.max(), 50)
contour = ax.contourf(X, Y, frames[0].reshape(Nx, Ny), levels=levels)

def update(frame):
    ax.clear()
    contour = ax.contourf(X, Y, frame.reshape(Nx, Ny), levels=levels)
    return contour

anim = FuncAnimation(fig, update, frames=frames, interval=50)
plt.show()

