from functions import D1_left, D1_right
import numpy as np

class BoundaryConditions:
    def __init__(self, room, lambda_air=0.026, T_out=273):
        self.room = room
        self.lambda_air = lambda_air
        self.T_out = T_out

    def modify_matrix(self, A):
        Nx, Ny = self.room.nx, self.room.ny
        hx, hy = self.room.hx, self.room.hy

        id_x, id_y = np.eye(Nx), np.eye(Ny)

        D1_left_x, D1_right_x = D1_left(Nx), D1_right(Nx)
        D1_left_y, D1_right_y = D1_left(Ny), D1_right(Ny)

        I = np.eye(Nx * Ny)

        for name, (mask, lam) in self.room.walls.items():
            beta = lam / self.lambda_air

            if "left" in name:
                A[mask, :] = (1/hx * np.kron(id_y, D1_left_x) + beta * I)[mask, :]

            if "right" in name:
                A[mask, :] = (1/hx * np.kron(id_y, D1_right_x) + beta * I)[mask, :]

            if "bottom" in name:
                A[mask, :] = (1/hy * np.kron(D1_left_y, id_x) + beta * I)[mask, :]

            if "top" in name:
                A[mask, :] = (1/hy * np.kron(D1_right_y, id_x) + beta * I)[mask, :]

        for name, (mask, lam) in self.room.windows.items():
            beta = lam / self.lambda_air

            if "left" in name:
                A[mask, :] = (1 / hx * np.kron(id_y, D1_left_x) + beta * I)[mask, :]

            if "right" in name:
                A[mask, :] = (1 / hx * np.kron(id_y, D1_right_x) + beta * I)[mask, :]

            if "bottom" in name:
                A[mask, :] = (1 / hy * np.kron(D1_left_y, id_x) + beta * I)[mask, :]

            if "top" in name:
                A[mask, :] = (1 / hy * np.kron(D1_right_y, id_x) + beta * I)[mask, :]

        return A

    def modify_rhs(self, rhs):
        for name, (mask, lam) in self.room.walls.items():
            beta = lam/self.lambda_air
            rhs[mask] = beta * self.T_out

        for name, (mask, lam) in self.room.windows.items() :
            beta = lam / self.lambda_air
            rhs[mask] = beta * self.T_out

        return rhs