### `src/utils.py`
```python
import numpy as np
import csv

def load_movielens(path="u.data"):
    data = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile, delimiter="\t")
        for row in reader:
            data.append([int(row[0])-1, int(row[1])-1, int(row[2])])
    data = np.array(data)

    num_obs = len(data)
    num_users = max(data[:,0]) + 1
    num_items = max(data[:,1]) + 1

    np.random.seed(1)
    num_train = int(0.8 * num_obs)
    perm = np.random.permutation(num_obs)

    train = data[perm[:num_train], :]
    test = data[perm[num_train:], :]
    return train, test, num_users, num_items

def mse(R_hat, dataset):
    error = 0.0
    for user, movie, rating in dataset:
        pred = R_hat[movie, user]
        error += (pred - rating) ** 2
    return error / len(dataset)