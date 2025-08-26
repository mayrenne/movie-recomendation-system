# 🎬 Movie Recommendation System (Matrix Completion)

This project builds a **personalized movie recommendation system** using the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/).  
The system applies **matrix completion** methods, including **SVD** and **alternating minimization**, to predict user ratings and recommend movies.


## 📂 Project Structure
- `src/`
  - `svd_baseline.py` → Implements rank-d SVD approximation.
  - `matrix_factorization.py` → Alternating minimization with regularization.
  - `utils.py` → Data loading, train/test split, and error calculation.
- `notebooks/`
  - `recommendation_demo.ipynb` → Interactive notebook with plots & sample results.
- `results/`
  - Saved outputs.
