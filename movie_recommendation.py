# -*- coding: utf-8 -*-
"""
Matrix Completion and Recommendation System
MovieLens 100k dataset

This script implements:
- Rank-one estimator (average movie ratings)
- Rank-d SVD approximation
- Alternating minimization with regularization for latent factors
"""

import csv
import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

# ----------------------------
# Step 1: Load the dataset
# ----------------------------
def load_movielens_data(filename='u.data', train_ratio=0.8, seed=1):
    data = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            data.append([int(row[0])-1, int(row[1])-1, int(row[2])])
    data = np.array(data)

    num_observations = len(data)
    num_users = max(data[:,0])+1
    num_items = max(data[:,1])+1

    np.random.seed(seed)
    num_train = int(train_ratio * num_observations)
    perm = np.random.permutation(num_observations)
    train = data[perm[:num_train], :]
    test = data[perm[num_train:], :]

    print(f"Loaded MovieLens 100k: {len(train)} train, {len(test)} test samples")
    return train, test, num_users, num_items

# ----------------------------
# Step 2: Rank-one estimator
# ----------------------------
def rank_one_estimator(train, num_users, num_items):
    movie_sum = np.zeros(num_items)
    movie_count = np.zeros(num_items)

    for user, movie, rating in train:
        movie_sum[movie] += rating
        movie_count[movie] += 1

    mu = np.divide(movie_sum, movie_count, out=np.zeros_like(movie_sum), where=movie_count!=0)
    R_hat = mu[:, None] * np.ones((1, num_users))
    return R_hat

def compute_mse(R_hat, dataset):
    error = 0.0
    for user, movie, rating in dataset:
        error += (R_hat[movie, user] - rating) ** 2
    return error / len(dataset)

# ----------------------------
# Step 3: Rank-d SVD estimator
# ----------------------------
def svd_estimator(r_twiddle, d):
    U, sigma, Vt = svds(r_twiddle, k=d)
    return U @ np.diag(sigma) @ Vt

def evaluate_svd(r_twiddle, train, test, d_values=[1,2,5,10,20,50]):
    train_errors, test_errors = [], []
    for d in d_values:
        R_hat = svd_estimator(r_twiddle, d)
        train_errors.append(compute_mse(R_hat, train))
        test_errors.append(compute_mse(R_hat, test))
    return train_errors, test_errors

# ----------------------------
# Step 4: Alternating Minimization
# ----------------------------
def closed_form_u(V, U, l, r_twiddle):
    d = U.shape[1]
    I = np.identity(d)
    m, n = r_twiddle.shape
    for i in range(m):
        users_who_rated_i = np.where(r_twiddle[i, :] != 0)[0]
        V_sum = np.dot(V[users_who_rated_i].T, V[users_who_rated_i])
        Rv_sum = np.dot(r_twiddle[i, users_who_rated_i], V[users_who_rated_i])
        U[i] = np.linalg.solve(V_sum + l * I, Rv_sum)
    return U

def closed_form_v(V, U, l, r_twiddle):
    d = U.shape[1]
    I = np.identity(d)
    m, n = r_twiddle.shape
    for j in range(n):
        movies_rated_by_j = np.where(r_twiddle[:, j] != 0)[0]
        U_sum = np.dot(U[movies_rated_by_j].T, U[movies_rated_by_j])
        Rv_sum = np.dot(r_twiddle[movies_rated_by_j, j], U[movies_rated_by_j])
        V[j] = np.linalg.solve(U_sum + l * I, Rv_sum)
    return V

def alternating_minimization(r_twiddle, d, l=10.0, delta=1e-1, max_iters=50):
    m, n = r_twiddle.shape
    U = np.random.randn(m, d)
    V = np.random.randn(n, d)
    
    prev_loss = float('inf')
    for iteration in range(max_iters):
        U = closed_form_u(V, U, l, r_twiddle)
        V = closed_form_v(V, U, l, r_twiddle)
        R_hat = np.dot(U, V.T)
        observed = r_twiddle != 0
        current_loss = np.sum((R_hat[observed] - r_twiddle[observed])**2) + \
                       l * (np.linalg.norm(U)**2 + np.linalg.norm(V)**2)
        if prev_loss - current_loss < delta:
            break
        prev_loss = current_loss
    return U, V

def compute_latent_mse(U, V, dataset):
    error = 0.0
    for user, movie, rating in dataset:
        error += (np.dot(U[movie], V[user]) - rating) ** 2
    return error / len(dataset)

# ----------------------------
# Step 5: Main execution
# ----------------------------
if __name__ == "__main__":
    train, test, num_users, num_items = load_movielens_data()

    # Rank-one estimator
    R_hat = rank_one_estimator(train, num_users, num_items)
    print("Rank-one Test MSE:", compute_mse(R_hat, test))

    # Prepare matrix for SVD / Alternating Minimization
    r_twiddle = np.zeros((num_items, num_users))
    for user, movie, rating in train:
        r_twiddle[movie, user] = rating

    # Evaluate SVD estimators
    d_values = [1,2,5,10,20,50]
    train_svd, test_svd = evaluate_svd(r_twiddle, train, test, d_values)
    plt.figure(figsize=(8,6))
    plt.plot(d_values, train_svd, label='Train MSE (SVD)')
    plt.plot(d_values, test_svd, label='Test MSE (SVD)')
    plt.xlabel('Latent factors d')
    plt.ylabel('MSE')
    plt.title('SVD Train/Test MSE')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Alternating Minimization
    train_alt, test_alt = [], []
    for d in d_values:
        U, V = alternating_minimization(r_twiddle, d)
        train_alt.append(compute_latent_mse(U, V, train))
        test_alt.append(compute_latent_mse(U, V, test))
    plt.figure(figsize=(8,6))
    plt.plot(d_values, train_alt, label='Train MSE (Alt)')
    plt.plot(d_values, test_alt, label='Test MSE (Alt)')
    plt.xlabel('Latent factors d')
    plt.ylabel('MSE')
    plt.title('Alternating Minimization Train/Test MSE')
    plt.legend()
    plt.grid(True)
    plt.show()
