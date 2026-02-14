import numpy as np
import matplotlib.pyplot as plt
import tqdm
from room import Room
from heater import Heater
from heatercontroller import HeaterController
from boundaryconditions import BoundaryConditions
from heatsolver import HeatSolver

T_max = 3600.0
steps = np.arange(0.05, 5.01, 0.05)
Nx, Ny = 20, 20
room_w, room_l = 2.0, 2.0
hx, hy = room_w / (Nx - 1), room_l / (Ny - 1)
T_initial = 293.0

def setup_simulation():
    r = Room(Nx, Ny, hx, hy, 0, 0, room_w, room_l)

    x_coords = np.linspace(0, room_w, Nx)
    y_coords = np.linspace(0, room_l, Ny)
    X, Y = np.meshgrid(x_coords, y_coords)
    X_f, Y_f = X.flatten(), Y.flatten()

    ind_win = (Y_f == y_coords[-1]) & (X_f >= 0.5) & (X_f <= 1.5)
    r.add_window("top", ind_win, 0.9 * hx)

    r.add_walls("left", X_f == x_coords[0], 0.2 * hx)
    r.add_walls("right", X_f == x_coords[-1], 0.2 * hx)
    r.add_walls("bottom", Y_f == y_coords[0], 0.2 * hx)
    r.add_walls("top", (Y_f == y_coords[-1]) & ~ind_win, 0.2 * hx)

    offset = 2 * hy
    ind_heat = (Y_f <= room_l - offset) & (Y_f > room_l - offset - 3 * hy) & (X_f >= 0.5) & (X_f <= 1.5)
    r.add_heater("heater_top", Heater(ind_heat, 915.0, 295.0))

    return r


def run_full_sim(ht, room_obj):
    solver = HeatSolver(room_obj, ht)
    bc = BoundaryConditions(room_obj)
    controller = HeaterController(room_obj)

    solver.A = solver.A.tolil()
    solver.A = bc.modify_matrix(solver.A).tocsc()
    from scipy.sparse.linalg import factorized
    solver.solver = factorized(solver.A)

    u_current = np.ones(Nx * Ny) * T_initial
    Nt = int(round(T_max / ht))
    history = np.zeros((Nt + 1, Nx * Ny))
    history[0] = u_current

    for n in range(1, Nt + 1):
        source = controller.compute_source(u_current)
        u_current = solver.step(u_current, source, bc)
        history[n] = u_current

    return history


errors_mean = []
errors_last = []
room_ref = setup_simulation()

print("Analiza błędów (Logika: Room 1, T=3600s)...")
for ht in tqdm.tqdm(steps):
    U_h = run_full_sim(ht, room_ref)

    U_h2 = run_full_sim(ht / 2, room_ref)

    min_len = min(len(U_h), len(U_h2[::2]))
    diff = np.linalg.norm(U_h[:min_len] - U_h2[::2][:min_len], axis=1)

    errors_mean.append(np.mean(diff))
    errors_last.append(diff[-1])

plt.plot(steps, errors_mean, 'b-')
plt.yscale('log')
plt.title("Błąd średni (Globalny)")
plt.xlabel("Krok czasowy ht [s]")

plt.show()