from pipelines.functions import D1_backward, D1_forward
import numpy as np


class BoundaryConditions:
    '''Klasa nakladajaca warunki brzegowe Robina na macierz A oraz wektor temperatur'''
    def __init__(self, room, lambda_air=0.026, u_ext=263.0):
        self.room = room
        self.lambda_air = lambda_air
        self.u_ext = u_ext

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

    def modify_matrix(self, A):
        '''Funkcja sluzy do nalozenia na macierz A warunkow brzegowych Robina'''
        id_y = np.eye(self.room.nx)
        id_x = np.eye(self.room.ny)
        I = np.eye(self.room.nx * self.room.ny)

        beta_wall = self.room.walls["left"][1]/self.lambda_air
        beta_window = self.lambda_window/self.lambda_air

        Bx_forward = -np.kron(id_y, D1_forward(self.room.nx)) / self.room.hx + I * beta_wall
        Bx_backward = np.kron(id_y, D1_backward(self.room.nx)) / self.room.hx + I * beta_wall
        By_forward = -np.kron(D1_forward(self.room.ny), id_x) / self.room.hy + I * beta_wall
        By_backward = np.kron(D1_backward(self.room.ny), id_x) / self.room.hy + I * beta_wall
        By_backward_window = np.kron(D1_backward(self.room.ny), id_x) / self.room.hy + I * beta_window

        A[self.ind_left, :] = Bx_forward[self.ind_left, :]
        A[self.ind_right, :] = Bx_backward[self.ind_right, :]
        A[self.ind_bottom, :] = By_forward[self.ind_bottom, :]
        A[self.ind_top, :] = By_backward[self.ind_top, :]
        A[self.ind_window, :] = By_backward_window[self.ind_window, :]

        return A

    def modify_rhs(self, rhs):
        '''Funkcja naklada na wektor temperatur warunki brzegowe Robina'''
        rhs = rhs.copy()
        beta_wall = self.room.walls["left"][1] / self.lambda_air
        beta_window = self.lambda_window / self.lambda_air

        rhs[self.ind_left] = beta_wall * self.u_ext
        rhs[self.ind_right] = beta_wall * self.u_ext
        rhs[self.ind_bottom] = beta_wall * self.u_ext
        rhs[self.ind_top] = beta_wall * self.u_ext
        rhs[self.ind_window] = beta_window * self.u_ext
        return rhs