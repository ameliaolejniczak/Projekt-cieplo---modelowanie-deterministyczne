from pipelines.functions import D1_backward, D1_forward
import numpy as np


class BoundaryConditions:
    '''Klasa nakladajaca warunki brzegowe Robina na macierz A oraz wektor temperatur'''
    def __init__(self, room, lambda_air=0.026, u_ext=263.0):
        self.room = room
        self.lambda_air = lambda_air
        self.u_ext = u_ext
        self.room.bc = self

        self.ind_left = self.room.walls["left"][0]
        self.ind_right = self.room.walls["right"][0]
        self.ind_bottom = self.room.walls["bottom"][0]
        self.ind_top = self.room.walls["top"][0]

        if "left" in self.room.windows.keys():
            self.ind_window = self.room.windows["left"][0]
            self.lambda_window = self.room.windows["left"][1]
        if "right" in self.room.windows.keys():
            self.ind_window = self.room.windows["right"][0]
            self.lambda_window = self.room.windows["right"][1]
        if "bottom" in self.room.windows.keys():
            self.ind_window = self.room.windows["bottom"][0]
            self.lambda_window = self.room.windows["bottom"][1]
        if "top" in self.room.windows.keys():
            self.ind_window = self.room.windows["top"][0]
            self.lambda_window = self.room.windows["top"][1]

    def get_beta(self, side):
        '''Zwraca betę zewnętrzną lub wewnętrzną w zależności od sąsiedztwa'''
        if side in self.room.neighbors:
            _, lambda_inner = self.room.neighbors[side]
            return lambda_inner / self.lambda_air
        return self.room.walls[side][1] / self.lambda_air

    def modify_matrix(self, A):
        '''Funkcja sluzy do nalozenia na macierz A warunkow brzegowych Robina'''
        id_y = np.eye(self.room.nx)
        id_x = np.eye(self.room.ny)
        I = np.eye(self.room.nx * self.room.ny)

        beta_window = self.lambda_window/self.lambda_air

        Bx_forward = -np.kron(id_y, D1_forward(self.room.nx)) / self.room.hx + I * self.get_beta("left")
        Bx_backward = np.kron(id_y, D1_backward(self.room.nx)) / self.room.hx + I * self.get_beta("right")
        By_forward = -np.kron(D1_forward(self.room.ny), id_x) / self.room.hy + I * self.get_beta("bottom")
        By_backward = np.kron(D1_backward(self.room.ny), id_x) / self.room.hy + I * self.get_beta("top")
        By_backward_window = np.kron(D1_backward(self.room.ny), id_x) / self.room.hy + I * beta_window

        A[self.ind_left, :] = Bx_forward[self.ind_left, :]
        A[self.ind_right, :] = Bx_backward[self.ind_right, :]
        A[self.ind_bottom, :] = By_forward[self.ind_bottom, :]
        A[self.ind_top, :] = By_backward[self.ind_top, :]
        A[self.ind_window, :] = By_backward_window[self.ind_window, :]

        return A

    def modify_rhs(self, rhs):
        '''Funkcja naklada na wektor temperatur warunki brzegowe Robina'''
        for side in ["left", "right", "bottom", "top"]:
            beta = self.get_beta(side)
            current_mask = getattr(self, f"ind_{side}")

            if side in self.room.neighbors:
                neighbor_room, _ = self.room.neighbors[side]
                u_neighbor = neighbor_room.last_u

                opposite = {"left": "right", "right": "left", "bottom": "top", "top": "bottom"}
                neighbor_mask = getattr(neighbor_room.bc, f"ind_{opposite[side]}")

                rhs[current_mask] = beta * u_neighbor[neighbor_mask]
            else:
                rhs[current_mask] = beta * self.u_ext

        beta_window = self.lambda_window / self.lambda_air
        rhs[self.ind_window] = beta_window * self.u_ext
        return rhs