import numpy as np

def D1_forward(N):
    '''Funkcja zwraca macierz pierwszej pochodnej z lewej strony o rozmiarze NxN'''
    return np.eye(N, k = 1) - np.eye(N)

def D1_backward(N):
    '''Funkcja zwraca macierz pierwszej pochodnej z prawej strony o rozmiarze NxN'''
    return - np.eye(N, k = -1) + np.eye(N)

def D2(N):
    '''Funkcja zwraca macierz drugiej pochodnej o rozmiarze NxN'''
    return - 2 * np.eye(N) + np.eye(N, k = 1) + np.eye(N, k = -1)
