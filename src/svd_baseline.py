import numpy as np
from scipy.sparse.linalg import svds
from .utils import mse

def svd_estimator(r_twiddle, d):
    U, sigma, Vt = svds(r_twiddle, k=d)
    return U @ np.diag(sigma) @ Vt

def evaluate_svd(r_twiddle, train, test, d_values=[1,2,5,10,20,50]):
    train_errors, test_errors = [], []
    for d in d_values:
        R_hat = svd_estimator(r_twiddle, d)
        train_errors.append(mse(R_hat, train))
        test_errors.append(mse(R_hat, test))
    return train_errors, test_errors