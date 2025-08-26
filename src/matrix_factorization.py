import numpy as np
from .utils import mse

def closed_form_u(V, U, l, r_twiddle):
    d = U.shape[1]
    I = np.identity(d)
    m, n = r_twiddle.shape
    for i in range(m):
        users = np.where(r_twiddle[i, :] != 0)[0]
        if len(users) == 0: continue
        V_sum = V[users].T @ V[users]
        Rv_sum = r_twiddle[i, users] @ V[users]
        U[i] = np.linalg.solve(V_sum + l*I, Rv_sum)
    return U

def closed_form_v(V, U, l, r_twiddle):
    d = U.shape[1]
    m, n = r_twiddle.shape
    I = np.identity(d)
    for j in range(n):
        movies = np.where(r_twiddle[:, j] != 0)[0]
        if len(movies) == 0: continue
        U_sum = U[movies].T @ U[movies]
        Rv_sum = r_twiddle[movies, j] @ U[movies]
        V[j] = np.linalg.solve(U_sum + l*I, Rv_sum)
    return V

def alternating_minimization(r_twiddle, d, l=10.0, delta=1e-2):
    m, n = r_twiddle.shape
    U = np.random.randn(m, d)
    V = np.random.randn(n, d)
    prev_loss = float("inf")

    while True:
        U = closed_form_u(V, U, l, r_twiddle)
        V = closed_form_v(V, U, l, r_twiddle)
        R_hat = U @ V.T
        observed = (r_twiddle != 0)
        squared_error = np.sum((R_hat[observed] - r_twiddle[observed]) ** 2)
        reg = l * (np.linalg.norm(U) ** 2 + np.linalg.norm(V) ** 2)
        loss = squared_error + reg
        if prev_loss - loss < delta:
            break
        prev_loss = loss
    return U, V
