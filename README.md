# Movie Recommendation System

## Overview
This project implements a personalized movie recommendation system using the MovieLens 100k dataset. The goal is to predict user ratings for movies based on historical ratings. The system demonstrates the following approaches:

- **Rank-one estimator**: Predicts each movie’s rating as the average of ratings from all users.
- **Rank-d SVD approximation**: Uses singular value decomposition for low-rank approximations of the rating matrix.
- **Alternating minimization with regularization**: Learns latent factor representations for users and movies, optimizing for mean squared error (MSE) only on observed ratings.

---

## Dataset
- **MovieLens 100k**: 100,000 ratings from 943 users on 1,682 movies.
- Each user rated at least 20 movies.
- Dataset source: [https://grouplens.org/datasets/movielens/100k/](https://grouplens.org/datasets/movielens/100k/)

---

## Skills / Technologies
- **Programming & Libraries**: Python, NumPy, SciPy, PyTorch, Matplotlib
- **Mathematics / ML Concepts**: Linear algebra, singular value decomposition, matrix factorization, latent factor models, regularization
- **Software Practices**: Modular code, function encapsulation, reproducible experiments

---
## Project Structure
- `README.md`→ youre here! the main project description :)
- `movie_recommendations` → main 
- `results/`
  - figures of MSE results