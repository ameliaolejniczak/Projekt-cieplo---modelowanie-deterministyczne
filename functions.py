import numpy as np

# macierze pochodnych 1 i 2 rzędu (potrzebne)
def D1_left(N):
    return np.eye(N, k = 1) - np.eye(N)

def D1_right(N):
    return np.eye(N) - np.eye(N, k = -1)

def D2(N):
    return np.eye(N, k = 1) + np.eye(N, k = -1) - 2 * np.eye(N)
