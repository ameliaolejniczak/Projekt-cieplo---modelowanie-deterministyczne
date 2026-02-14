import numpy as np
import matplotlib.pyplot as plt
import tqdm
from room import Room
from heater import Heater
from heatercontroller import HeaterController
from boundaryconditions import BoundaryConditions
from heatsolver import HeatSolver
import matplotlib.animation as animation

room_width = 2.0
room_length = 2.0

c = 1005.0
p = 101325.0
P = 915.0
r = 287.05

U_window = 0.9
U_wall = 0.2
lambda_air = 0.0262

ht = 1.0
T = 3600.0 * 24
t = np.arange(0, T, ht)

Nx, Ny = 20, 20
x = np.linspace(0, room_width, Nx)
y = np.linspace(0, room_length, Ny)
hx, hy = x[1] - x[0], y[1] - y[0]
room1 = Room(Nx, Ny, hx, hy, 0, 0, room_width, room_length)
room2 = Room(Nx, Ny, hx, hy, 0, 0, room_width, room_length)
room3 = Room(Nx, Ny, hx, hy, 0, 0, room_width, room_length)
lambda_wall = U_wall * hx
lambda_window = U_window * hx

X, Y = np.meshgrid(x, y)
X_flat, Y_flat = X.flatten(), Y.flatten()

T_out = 263.0
T_initial = 293.0
T_target = 295.0

window_width = 1.0
window_center = room_width / 2
window_x_min = window_center - window_width / 2
window_x_max = window_center + window_width / 2
ind_window = (Y_flat == y[-1]) & (X_flat >= window_x_min) & (X_flat <= window_x_max)
room1.add_window("top", ind_window, lambda_window)
room2.add_window("top", ind_window, lambda_window)
room3.add_window("top", ind_window, lambda_window)

ind_left = np.where(X_flat == x[0], True, False)
ind_right = np.where(X_flat == x[-1], True, False)
ind_bottom = np.where(Y_flat == y[0], True, False)
ind_top = np.where(Y_flat == y[-1], True, False) & ~ind_window
room1.add_walls("left", ind_left, lambda_wall)
room1.add_walls("right", ind_right, lambda_wall)
room1.add_walls("bottom", ind_bottom, lambda_wall)
room1.add_walls("top", ind_top, lambda_wall)
room2.add_walls("left", ind_left, lambda_wall)
room2.add_walls("right", ind_right, lambda_wall)
room2.add_walls("bottom", ind_bottom, lambda_wall)
room2.add_walls("top", ind_top, lambda_wall)
room3.add_walls("left", ind_left, lambda_wall)
room3.add_walls("right", ind_right, lambda_wall)
room3.add_walls("bottom", ind_bottom, lambda_wall)
room3.add_walls("top", ind_top, lambda_wall)

radiator_width = 1.0
offset = 2 * hy
rad_x_min = window_center - radiator_width / 2
rad_x_max = window_center + radiator_width / 2

radiator_height = radiator_width
rad_y_center = room_length / 2
rad_y_min = rad_y_center - radiator_height / 2
rad_y_max = rad_y_center + radiator_height / 2

ind_radiator_top = (Y_flat <= room_width - offset) & (Y_flat > room_width - offset - 3 * hy) & (X_flat >= rad_x_min) & (X_flat <= rad_x_max)
heater_top = Heater(ind_radiator_top, P, T_target)
room1.add_heater("heater", heater_top)
ind_radiator_bottom = (Y_flat >= offset) & (Y_flat < offset + 3 * hy) & (X_flat >= rad_x_min) & (X_flat <= rad_x_max)
heater_bottom = Heater(ind_radiator_bottom, P, T_target)
room2.add_heater("heater", heater_bottom)
ind_radiator_left = (X_flat >= offset) & (X_flat < offset + 3 * hx) & (Y_flat >= rad_y_min) & (Y_flat <= rad_y_max)
heater_left = Heater(ind_radiator_left, P, T_target)
room3.add_heater("heater", heater_left)

bc1 = BoundaryConditions(room1)
bc2 = BoundaryConditions(room2)
bc3 = BoundaryConditions(room3)
solver1 = HeatSolver(room1, ht)
solver1.apply_boundary_conditions(bc1)
solver2 = HeatSolver(room2, ht)
solver2.apply_boundary_conditions(bc2)
solver3 = HeatSolver(room3, ht)
solver3.apply_boundary_conditions(bc3)

controller1 = HeaterController(room1)
controller2 = HeaterController(room2)
controller3 = HeaterController(room3)

u0 = np.ones(Nx * Ny) * 290
u_current1 = np.zeros(len(u0))
u_current2 = np.zeros(len(u0))
u_current3 = np.zeros(len(u0))

energy_usage1 = []
energy_usage2 = []
energy_usage3 = []

history1 = []
history2 = []
history3 = []
save_interval = 60

for time in tqdm.tqdm(t):
    if time == t[0]:
        u_current1 = u0.copy()
        u_current2 = u0.copy()
        u_current3 = u0.copy()
    else:
        source1 = controller1.compute_source(u_current1)
        energy_usage1.append(np.sum(source1) * hx * hy)
        u_current1 = solver1.step(u_current1, source1, bc1)
        source2 = controller2.compute_source(u_current2)
        energy_usage2.append(np.sum(source2) * hx * hy)
        u_current2 = solver2.step(u_current2, source2, bc2)
        source3 = controller3.compute_source(u_current3)
        energy_usage3.append(np.sum(source3) * hx * hy)
        u_current3 = solver3.step(u_current3, source3, bc3)
        if int(time) % save_interval == 0:
            history1.append(u_current1.reshape(Nx, Ny).copy())
            history2.append(u_current2.reshape(Nx, Ny).copy())
            history3.append(u_current3.reshape(Nx, Ny).copy())

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
axes = [ax1, ax2, ax3]
histories = [history1, history2, history3]
titles = ["Grzejnik: Góra (Okno)", "Grzejnik: Dół", "Grzejnik: Lewo"]

vmin, vmax = 263, 305

levels = np.arange(269, 355, 5)

def update(frame):
    for i in range(3):
        axes[i].clear()
        cont = axes[i].contourf(x, y, histories[i][frame], levels=levels)
        axes[i].set_title(f"{titles[i]}\nMin: {histories[i][frame].min()-273:.1f}°C")
        if frame == 0 and i == 2: # Dodaj pasek koloru tylko raz
            fig.colorbar(cont, ax=ax3)
    fig.suptitle(f"Czas symulacji: {frame * save_interval // 60} min")

ani = animation.FuncAnimation(fig, update, frames=len(history1), interval=100, repeat=False)
plt.tight_layout()
plt.show()
