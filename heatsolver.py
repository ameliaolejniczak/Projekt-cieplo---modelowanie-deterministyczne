import numpy as np
from functions import D2

class HeatSolver:
    def __init__(self, room, alpha, ht):
        self.room = room
        self.alpha = alpha
        self.ht = ht
        self.n = room.nx * room.ny

        self.I = np.eye(self.n)
        self.L = self.build_laplacian()
        self.A = self.I - ht * alpha * self.L

    def build_laplacian(self):
        Nx, Ny = self.room.nx, self.room.ny
        hx, hy = self.room.hx, self.room.hy

        id_x = np.eye(Nx)
        id_y = np.eye(Ny)

        D2_x = D2(Nx)
        D2_y = D2(Ny)

        return np.kron(id_y, D2_x) / hx**2 + np.kron(D2_y, id_x) / hy**2

    def apply_boundary_conditions(self, bc):
        self.A = bc.modify_matrix(self.A)

    def step(self, u, source, bc):
        rhs = u + self.ht * source * u
        rhs = bc.modify_rhs(rhs)
        return np.linalg.solve(self.A, rhs)
