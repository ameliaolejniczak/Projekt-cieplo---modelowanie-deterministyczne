import numpy as np
import matplotlib.pyplot as plt
import tqdm

# macierze pochodnych 1 i 2 rzędu (potrzebne)
def D1_left(N):
    return np.eye(N, k = 1) - np.eye(N)

def D1_right(N):
    return np.eye(N) - np.eye(N, k = -1)

def D2(N):
    return np.eye(N, k = 1) + np.eye(N, k = -1) - 2 * np.eye(N)

# tworzenie macierzy ewolucji
# dyskretyzacja czasu (tak samo)
ht = 0.1
t = np.arange(0, 50.0, ht)

# dyskretyzacja przestrzeni (potrzebne)
Nx, Ny = 30, 30
x, y = np.linspace(-1, 1, Nx), np.linspace(-1, 1, Ny)
# x, y = np.linspace(0, 3, Nx), np.linspace(0, 3, Ny)
X, Y = np.meshgrid(x, y)
X_flat, Y_flat = X.flatten(), Y.flatten()
hx, hy = x[1] - x[0], y[1] - y[0]

# pierwsza przykładowa pozycja grzejnika
heater = np.where((Y_flat < -0.6) & (Y_flat > -0.8) & (X_flat > -0.3) & (X_flat < 0.3), True, False)
window = np.where((Y_flat == -1) & (X_flat > -0.3) & (X_flat < 0.3), True, False)

# heater = ((Y_flat < 0.05) & (Y_flat > 0.15) & (X_flat > 1.5) & (X_flat < 2.5))
# window = ((Y_flat == y[0]) & (X_flat > 1) & (X_flat < 3))


# identyfikacja punktów brzegowych w spłaszczonych wektorach przestrzeni
ind_brzeg_left = np.where(X_flat == x[0], True, False)
ind_brzeg_right = np.where(X_flat == x[-1], True, False)
ind_brzeg_bottom_wall = np.where((Y_flat == y[0]) & ((X_flat < -0.3) | (X_flat > 0.3)), True, False)
# ind_brzeg_bottom_wall = np.where((Y_flat == y[0]) & ((X_flat < 1) | (X_flat > 3)), True, False)
ind_brzeg_top = np.where(Y_flat == y[-1], True, False)

# wygenerowanie macierzy identyczności i różniczkowania
id_x = np.eye(Nx)
id_y = np.eye(Ny)
I = np.eye(Nx * Ny)

D1_left_x = D1_left(Nx)
D1_left_y = D1_left(Ny)
D1_right_x = D1_right(Nx)
D1_right_y = D1_right(Ny)
D2_x = D2(Nx)
D2_y = D2(Ny)

# generujemy laplasjan
L = np.kron(id_y, D2_x) / hx**2 + np.kron(D2_y, id_x) / hy**2

#generujemy macierz ewolucji
alpha = 0.0684
A = I - ht * alpha * L

lambda_mat = 0.6   # przewodność ściany
lambda_air = 0.026    # powietrze
lambda_window = 0.96
beta = lambda_mat / lambda_air
beta_window = lambda_window / lambda_air
u_air = 273.0         # temperatura otoczenia

# dodajemy warunki brzegowe
A[ind_brzeg_left, :] = (1/hx * np.kron(id_y, D1_left_x) + beta * I)[ind_brzeg_left, :]
A[ind_brzeg_right, :] = (1/hx * np.kron(id_y, D1_right_x) + beta * I)[ind_brzeg_right, :]
A[window, :] = (1/hy * np.kron(D1_left_y, id_x) + beta_window * I)[window, :]
A[ind_brzeg_bottom_wall, :] = (1/hy * np.kron(D1_left_y, id_x) + beta * I)[ind_brzeg_bottom_wall, :]
A[ind_brzeg_top, :] = (1/hy * np.kron(D1_right_y, id_x) + beta * I)[ind_brzeg_top, :]
# A[window, :] = (1/hy * np.kron(D1_right_y, id_x) + beta_window * I)[window, :]

u0 = np.ones(len(X_flat)) * 292

u_current = np.zeros(len(u0))
P = 915.0  # "moc" grzejnika – skalowanie
p = 101325.0 # ciśnienie powietrza w pascalach
r = 287.05 # J/(kg K)
c = 1005 # ciepło właściwe suchego powietrza
A_heater = 0.6
f = np.zeros_like(u_current)
f[heater] = 3600 * P * r / (p * A_heater * c)
print(f[heater].mean())

indices = ind_brzeg_left | ind_brzeg_right | ind_brzeg_top | ind_brzeg_bottom_wall

for time in tqdm.tqdm(t):
    if time == t[0]:
        u_current = u0.copy()
    else:
        f_eff = f * (u_current < 298) * u_current
        rhs = u_current + ht * f_eff
        rhs[indices] = beta * u_air
        rhs[window] = beta_window * u_air
        u_current = np.linalg.solve(A, rhs)

levels = np.linspace(u_current.min(), u_current.max(), 50)

plt.contourf(X, Y, u_current.reshape(Nx, Ny), levels = levels)
plt.title("Wynik końcowy")
plt.colorbar()

plt.show()

print(max(u_current))
print(u_current.mean())

