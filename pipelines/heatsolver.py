import numpy as np
from functions import D2
from scipy.sparse import eye, kron, csc_matrix
from scipy.sparse.linalg import factorized

class HeatSolver:
    '''Klasa rozwiazujaca rownanie ciepla'''
    def __init__(self, room, ht, alpha = 2.239 * 10 **(-5)):
        self.room = room
        self.alpha = alpha
        self.ht = ht
        self.n = room.nx * room.ny

        self.I = eye(self.n, format = "csc")
        self.L = self.build_laplacian()
        self.A = (self.I - ht * alpha * self.L)


    def build_laplacian(self):
        '''Funkcja zwracajaca laplasjan'''
        Nx, Ny = self.room.nx, self.room.ny
        hx, hy = self.room.hx, self.room.hy

        id_x = np.eye(Nx)
        id_y = np.eye(Ny)

        D2_x = csc_matrix(D2(Nx))
        D2_y = csc_matrix(D2(Ny))

        return kron(id_y, D2_x) / hx**2 + kron(D2_y, id_x) / hy**2

    def apply_boundary_conditions(self, bc):
        '''Funkcja zwraca macierz A po nalozeniu na nia warunkow brzegowych
        oraz przygotowuje funkcje do rozwiazania rownania Ax = b'''
        self.A = bc.modify_matrix(self.A).tocsc()
        self.solver = factorized(self.A)

    def step(self, u, source, bc):
        '''Funkcja przeprowadza jeden krok czasowy'''
        rhs = u + self.ht * source
        rhs = bc.modify_rhs(rhs)
        return self.solver(rhs)
