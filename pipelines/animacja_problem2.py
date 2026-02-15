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
U_wall_inside = 1.0
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
lambda_wall_inside = U_wall_inside * hx
lambda_window = U_window * hx

X, Y = np.meshgrid(x, y)
X_flat, Y_flat = X.flatten(), Y.flatten()

T_out = 263.0
T_initial = 293.0
T_target = 298.0

window_width = 1.0
window_center = room_width / 2
window_x_min = window_center - window_width / 2
window_x_max = window_center + window_width / 2
ind_window1 = np.where((X_flat == x[0]) & (Y_flat >= window_x_min) & (Y_flat <= window_x_max), True, False)
ind_window2 = np.where((Y_flat == y[-1]) & (X_flat >= window_x_min) & (X_flat <= window_x_max), True, False)
ind_window3 = np.where((X_flat == x[-1]) & (Y_flat >= window_x_min) & (Y_flat <= window_x_max), True, False)
room1.add_window("top", ind_window1, lambda_window)
room2.add_window("top", ind_window2, lambda_window)
room3.add_window("top", ind_window3, lambda_window)

ind_left = np.where(X_flat == x[0], True, False)
ind_left1 = np.where(X_flat == x[0], True, False) & ~ind_window1
ind_right = np.where(X_flat == x[-1], True, False)
ind_right3 = np.where(X_flat == x[-1], True, False) & ~ind_window3
ind_bottom = np.where(Y_flat == y[0], True, False)
ind_top = np.where(Y_flat == y[-1], True, False)
ind_top2 = np.where(Y_flat == y[-1], True, False) & ~ind_window2
room1.add_walls("left", ind_left1, lambda_wall)
room1.add_walls("right", ind_right, lambda_wall_inside)
room1.add_walls("bottom", ind_bottom, lambda_wall)
room1.add_walls("top", ind_top, lambda_wall)
room2.add_walls("left", ind_left, lambda_wall_inside)
room2.add_walls("right", ind_right, lambda_wall_inside)
room2.add_walls("bottom", ind_bottom, lambda_wall)
room2.add_walls("top", ind_top2, lambda_wall)
room3.add_walls("left", ind_left, lambda_wall_inside)
room3.add_walls("right", ind_right3, lambda_wall)
room3.add_walls("bottom", ind_bottom, lambda_wall)
room3.add_walls("top", ind_top, lambda_wall)

room1.add_neighbor("right", room2)
room2.add_neighbor("left", room1)
room2.add_neighbor("right", room3)
room3.add_neighbor("left", room2)

radiator_width = 1.0
offset = 2 * hy
rad_x_min = window_center - radiator_width / 2
rad_x_max = window_center + radiator_width / 2

radiator_height = radiator_width
rad_y_center = room_length / 2
rad_y_min = rad_y_center - radiator_height / 2
rad_y_max = rad_y_center + radiator_height / 2

ind_radiator_left = (X_flat >= offset) & (X_flat < offset + 3 * hx) & (Y_flat >= rad_y_min) & (Y_flat <= rad_y_max)
heater_left = Heater(ind_radiator_left, P, T_target)
room1.add_heater("heater", heater_left)
ind_radiator_right = (X_flat <= room_length - offset) & (X_flat > room_length - offset - 3 * hx) & (Y_flat >= rad_y_min) & (Y_flat <= rad_y_max)
heater_right = Heater(ind_radiator_right, P, T_target)
room3.add_heater("heater", heater_right)

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

room1.last_u = u0.copy()
room2.last_u = u0.copy()
room3.last_u = u0.copy()

energy_usage1, energy_usage2, energy_usage3 = [], [], []
history1, history2, history3 = [], [], []
save_interval = 60

for time in tqdm.tqdm(t):
    source1 = controller1.compute_source(room1.last_u)
    source2 = controller2.compute_source(room2.last_u)
    source3 = controller3.compute_source(room3.last_u)

    energy_usage1.append(np.sum(source1) * hx * hy)
    energy_usage2.append(np.sum(source2) * hx * hy)
    energy_usage3.append(np.sum(source3) * hx * hy)

    u_next1 = solver1.step(room1.last_u, source1, bc1)
    u_next2 = solver2.step(room2.last_u, source2, bc2)
    u_next3 = solver3.step(room3.last_u, source3, bc3)

    room1.last_u = u_next1
    room2.last_u = u_next2
    room3.last_u = u_next3

    if int(time) % save_interval == 0:
        history1.append(room1.last_u.reshape(Nx, Ny).copy())
        history2.append(room2.last_u.reshape(Nx, Ny).copy())
        history3.append(room3.last_u.reshape(Nx, Ny).copy())

u_current1 = room1.last_u - 273.0
u_current2 = room2.last_u - 273.0
u_current3 = room3.last_u - 273.0

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
results = [u_current1, u_current2, u_current3]
titles = ["Pokój pierwszy", "Pokój drugi", "Pokój trzeci"]

vmin = min(u.min() for u in results)
vmax = max(u.max() for u in results)

for i, ax in enumerate(axes):
    u_2d = results[i].reshape(Nx, Ny)

    im = ax.contourf(x, y, u_2d, levels=30, vmin=vmin, vmax=vmax)
    ax.set_title(titles[i])
    ax.set_xlabel("Szerokość [m]")
    ax.set_ylabel("Długość [m]")

    print(f"\n--- Statystyki: {titles[i]} ---")
    print(f"Średnia: {np.mean(results[i]):.2f}°C")
    print(f"Sigma (std): {np.std(results[i]):.2f}°C")
    print(f"Max: {results[i].max():.2f}°C | Min: {results[i].min():.2f}°C")

fig.colorbar(im, ax=axes.ravel().tolist(), label="Temperatura [°C]")
plt.suptitle("Porównanie rozkładu temperatury trzech sąsiadów", fontsize=16)

plt.show()
print(f"Zużycie energii w pierwszym pokoju: {np.sum(energy_usage1) * ht}")
print(f"Zużycie energii w drugim pokoju: {np.sum(energy_usage2) * ht}")
print(f"Zużycie energii w trzecim pokoju: {np.sum(energy_usage3) * ht}")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
axes = [ax1, ax2, ax3]
histories = [history1, history2, history3]
titles = ["Pokój pierwszy", "Pokój drugi", "Pokój trzeci"]

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


room_width = 2.0
room_length = 2.0

c = 1005.0
p = 101325.0
P = 915.0
r = 287.05

U_window = 0.9
U_wall = 0.2
U_wall_inside = 1.0
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
lambda_wall_inside = U_wall_inside * hx
lambda_window = U_window * hx

X, Y = np.meshgrid(x, y)
X_flat, Y_flat = X.flatten(), Y.flatten()

T_out = 263.0
T_initial = 293.0
T_target = 298.0

window_width = 1.0
window_center = room_width / 2
window_x_min = window_center - window_width / 2
window_x_max = window_center + window_width / 2
ind_window1 = np.where((X_flat == x[0]) & (Y_flat >= window_x_min) & (Y_flat <= window_x_max), True, False)
ind_window2 = np.where((Y_flat == y[-1]) & (X_flat >= window_x_min) & (X_flat <= window_x_max), True, False)
ind_window3 = np.where((X_flat == x[-1]) & (Y_flat >= window_x_min) & (Y_flat <= window_x_max), True, False)
room1.add_window("top", ind_window1, lambda_window)
room2.add_window("top", ind_window2, lambda_window)
room3.add_window("top", ind_window3, lambda_window)

ind_left = np.where(X_flat == x[0], True, False)
ind_left1 = np.where(X_flat == x[0], True, False) & ~ind_window1
ind_right = np.where(X_flat == x[-1], True, False)
ind_right3 = np.where(X_flat == x[-1], True, False) & ~ind_window3
ind_bottom = np.where(Y_flat == y[0], True, False)
ind_top = np.where(Y_flat == y[-1], True, False)
ind_top2 = np.where(Y_flat == y[-1], True, False) & ~ind_window2
room1.add_walls("left", ind_left1, lambda_wall)
room1.add_walls("right", ind_right, lambda_wall_inside)
room1.add_walls("bottom", ind_bottom, lambda_wall)
room1.add_walls("top", ind_top, lambda_wall)
room2.add_walls("left", ind_left, lambda_wall_inside)
room2.add_walls("right", ind_right, lambda_wall_inside)
room2.add_walls("bottom", ind_bottom, lambda_wall)
room2.add_walls("top", ind_top2, lambda_wall)
room3.add_walls("left", ind_left, lambda_wall_inside)
room3.add_walls("right", ind_right3, lambda_wall)
room3.add_walls("bottom", ind_bottom, lambda_wall)
room3.add_walls("top", ind_top, lambda_wall)

room1.add_neighbor("right", room2)
room2.add_neighbor("left", room1)
room2.add_neighbor("right", room3)
room3.add_neighbor("left", room2)

radiator_width = 1.0
offset = 2 * hy
rad_x_min = window_center - radiator_width / 2
rad_x_max = window_center + radiator_width / 2

radiator_height = radiator_width
rad_y_center = room_length / 2
rad_y_min = rad_y_center - radiator_height / 2
rad_y_max = rad_y_center + radiator_height / 2

ind_radiator_left = (X_flat >= offset) & (X_flat < offset + 3 * hx) & (Y_flat >= rad_y_min) & (Y_flat <= rad_y_max)
heater_left = Heater(ind_radiator_left, P, T_target)
room1.add_heater("heater", heater_left)
ind_radiator_right = (X_flat <= room_length - offset) & (X_flat > room_length - offset - 3 * hx) & (Y_flat >= rad_y_min) & (Y_flat <= rad_y_max)
heater_right = Heater(ind_radiator_right, P, T_target)
room3.add_heater("heater", heater_right)
ind_radiator_top = (Y_flat <= room_width - offset) & (Y_flat > room_width - offset - 3 * hy) & (X_flat >= rad_x_min) & (X_flat <= rad_x_max)
heater_top = Heater(ind_radiator_top, P, T_target)
room2.add_heater("heater", heater_top)

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

room1.last_u = u0.copy()
room2.last_u = u0.copy()
room3.last_u = u0.copy()

energy_usage1, energy_usage2, energy_usage3 = [], [], []
history1, history2, history3 = [], [], []
save_interval = 60

for time in tqdm.tqdm(t):
    source1 = controller1.compute_source(room1.last_u)
    source2 = controller2.compute_source(room2.last_u)
    source3 = controller3.compute_source(room3.last_u)

    energy_usage1.append(np.sum(source1) * hx * hy)
    energy_usage2.append(np.sum(source2) * hx * hy)
    energy_usage3.append(np.sum(source3) * hx * hy)

    u_next1 = solver1.step(room1.last_u, source1, bc1)
    u_next2 = solver2.step(room2.last_u, source2, bc2)
    u_next3 = solver3.step(room3.last_u, source3, bc3)

    room1.last_u = u_next1
    room2.last_u = u_next2
    room3.last_u = u_next3

    if int(time) % save_interval == 0:
        history1.append(room1.last_u.reshape(Nx, Ny).copy())
        history2.append(room2.last_u.reshape(Nx, Ny).copy())
        history3.append(room3.last_u.reshape(Nx, Ny).copy())

u_current1 = room1.last_u - 273.0
u_current2 = room2.last_u - 273.0
u_current3 = room3.last_u - 273.0

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
results = [u_current1, u_current2, u_current3]
titles = ["Pokój pierwszy", "Pokój drugi", "Pokój trzeci"]

vmin = min(u.min() for u in results)
vmax = max(u.max() for u in results)

for i, ax in enumerate(axes):
    u_2d = results[i].reshape(Nx, Ny)

    im = ax.contourf(x, y, u_2d, levels=30, vmin=vmin, vmax=vmax)
    ax.set_title(titles[i])
    ax.set_xlabel("Szerokość [m]")
    ax.set_ylabel("Długość [m]")

    print(f"\n--- Statystyki: {titles[i]} ---")
    print(f"Średnia: {np.mean(results[i]):.2f}°C")
    print(f"Sigma (std): {np.std(results[i]):.2f}°C")
    print(f"Max: {results[i].max():.2f}°C | Min: {results[i].min():.2f}°C")

fig.colorbar(im, ax=axes.ravel().tolist(), label="Temperatura [°C]")
plt.suptitle("Porównanie rozkładu temperatury trzech sąsiadów", fontsize=16)

plt.show()
print(f"Zużycie energii w pierwszym pokoju: {np.sum(energy_usage1) * ht}")
print(f"Zużycie energii w drugim pokoju: {np.sum(energy_usage2) * ht}")
print(f"Zużycie energii w trzecim pokoju: {np.sum(energy_usage3) * ht}")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
axes = [ax1, ax2, ax3]
histories = [history1, history2, history3]
titles = ["Pokój pierwszy", "Pokój drugi", "Pokój trzeci"]

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