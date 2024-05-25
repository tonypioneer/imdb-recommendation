import pandas as pd
import numpy as np
import os
import csv
import random
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# Load data
ratings = pd.read_csv('../data/input/ratings.csv')
usrid = []
movieid = []
for i in range(len(ratings['userId'])):
    if ratings['userId'][i] not in usrid:
        usrid.append(ratings['userId'][i])
    if ratings['movieId'][i] not in movieid:
        movieid.append(ratings['movieId'][i])

print(len(usrid))
print(len(movieid))

train = []
valid = []
data_all = []
index = 0
for user in usrid:
    this_user = []
    if index >= len(ratings['userId']):
        break
    while ratings['userId'][index] == user:
        temp = [ratings['userId'][index], ratings['movieId'][index], ratings['rating'][index]]
        this_user.append(temp)
        index += 1
        if index >= len(ratings['userId']):
            break
    print(len(this_user))
    data_all.append(this_user)

threshold = 0.85
test_data = []
with open("../data/output/train.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['userId', 'movieId', 'rating'])
    for this_user in data_all:
        length = len(this_user)
        for i in range(length):
            temp = random.random()
            if temp < threshold:
                writer.writerow(this_user[i])
            else:
                test_data.append(this_user[i])

with open("../data/output/test.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['userId', 'movieId', 'rating'])
    for row in test_data:
        writer.writerow(row)

train_data = pd.read_csv("../data/output/train.csv")
test_data = pd.read_csv("../data/output/test.csv")

class SVD_SGD:
    def __init__(self, n_factors, lr, reg, n_epochs):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs

    def fit(self, train_data):
        n_users = train_data['userId'].nunique()
        n_items = train_data['movieId'].nunique()
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_map = {user: i for i, user in enumerate(train_data['userId'].unique())}
        self.item_map = {item: i for i, item in enumerate(train_data['movieId'].unique())}
        for epoch in range(self.n_epochs):
            for row in train_data.itertuples():
                user, item, rating = row.userId, row.movieId, row.rating
                user_idx = self.user_map[user]
                item_idx = self.item_map[item]
                error = rating - self.predict(user_idx, item_idx)
                self.user_factors[user_idx] += self.lr * (error * self.item_factors[item_idx] - self.reg * self.user_factors[user_idx])
                self.item_factors[item_idx] += self.lr * (error * self.user_factors[user_idx] - self.reg * self.item_factors[item_idx])

    def predict(self, user, item):
        if user >= len(self.user_factors) or item >= len(self.item_factors):
            return np.mean(train_data['rating'])  # Return global mean if out of bounds
        return np.dot(self.user_factors[user], self.item_factors[item])

class SVD_SGLD:
    def __init__(self, n_factors, lr, reg, n_epochs, noise_scale):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.noise_scale = noise_scale

    def fit(self, train_data):
        n_users = train_data['userId'].nunique()
        n_items = train_data['movieId'].nunique()
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_map = {user: i for i, user in enumerate(train_data['userId'].unique())}
        self.item_map = {item: i for i, item in enumerate(train_data['movieId'].unique())}
        for epoch in range(self.n_epochs):
            for row in train_data.itertuples():
                user, item, rating = row.userId, row.movieId, row.rating
                user_idx = self.user_map[user]
                item_idx = self.item_map[item]
                error = rating - self.predict(user_idx, item_idx)
                noise = np.random.normal(0, self.noise_scale, self.n_factors)
                update_user = self.lr * (error * self.item_factors[item_idx] - self.reg * self.user_factors[user_idx]) + noise
                update_item = self.lr * (error * self.user_factors[user_idx] - self.reg * self.item_factors[item_idx]) + noise
                self.user_factors[user_idx] += np.nan_to_num(update_user)
                self.item_factors[item_idx] += np.nan_to_num(update_item)

    def predict(self, user, item):
        if user >= len(self.user_factors) or item >= len(self.item_factors):
            return np.mean(train_data['rating'])  # Return global mean if out of bounds
        return np.dot(self.user_factors[user], self.item_factors[item])

class SVD_SGHMC:
    def __init__(self, n_factors, lr, reg, n_epochs, friction, noise_scale):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.friction = friction
        self.noise_scale = noise_scale

    def fit(self, train_data):
        n_users = train_data['userId'].nunique()
        n_items = train_data['movieId'].nunique()
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_map = {user: i for i, user in enumerate(train_data['userId'].unique())}
        self.item_map = {item: i for i, item in enumerate(train_data['movieId'].unique())}
        for epoch in range(self.n_epochs):
            for row in train_data.itertuples():
                user, item, rating = row.userId, row.movieId, row.rating
                user_idx = self.user_map[user]
                item_idx = self.item_map[item]
                error = rating - self.predict(user_idx, item_idx)
                noise = np.random.normal(0, self.noise_scale, self.n_factors)
                update_user = self.lr * (error * self.item_factors[item_idx] - self.reg * self.user_factors[user_idx]) - self.friction * self.user_factors[user_idx] + noise
                update_item = self.lr * (error * self.user_factors[user_idx] - self.reg * self.item_factors[item_idx]) - self.friction * self.item_factors[item_idx] + noise
                self.user_factors[user_idx] += np.nan_to_num(update_user)
                self.item_factors[item_idx] += np.nan_to_num(update_item)

    def predict(self, user, item):
        if user >= len(self.user_factors) or item >= len(self.item_factors):
            return np.mean(train_data['rating'])  # Return global mean if out of bounds
        return np.dot(self.user_factors[user], self.item_factors[item])

def evaluate_model(model, test_data):
    predictions = []
    for row in test_data.itertuples():
        user = model.user_map.get(row.userId, -1)
        item = model.item_map.get(row.movieId, -1)
        if user == -1 or item == -1:
            predictions.append(np.mean(train_data['rating']))  # Fallback to global mean rating
        else:
            predictions.append(model.predict(user, item))
    true_ratings = test_data['rating'].values
    rmse = np.sqrt(mean_squared_error(true_ratings, predictions))
    return rmse

# Train and evaluate SVD with SGD
svd_sgd = SVD_SGD(n_factors=20, lr=0.005, reg=0.05, n_epochs=20)
svd_sgd.fit(train_data)
rmse_sgd = evaluate_model(svd_sgd, test_data)
print(f'RMSE for SVD with SGD: {rmse_sgd}')

# Train and evaluate SVD with SGLD
svd_sgld = SVD_SGLD(n_factors=20, lr=0.005, reg=0.05, n_epochs=20, noise_scale=0.01)
svd_sgld.fit(train_data)
rmse_sgld = evaluate_model(svd_sgld, test_data)
print(f'RMSE for SVD with SGLD: {rmse_sgld}')

# Train and evaluate SVD with SGHMC
svd_sghmc = SVD_SGHMC(n_factors=20, lr=0.005, reg=0.05, n_epochs=20, friction=0.05, noise_scale=0.01)
svd_sghmc.fit(train_data)
rmse_sghmc = evaluate_model(svd_sghmc, test_data)
print(f'RMSE for SVD with SGHMC: {rmse_sghmc}')

def visualize_clusters(model):
    pca = PCA(n_components=2)
    movie_factors_2d = pca.fit_transform(model.item_factors)
    plt.scatter(movie_factors_2d[:, 0], movie_factors_2d[:, 1], alpha=0.5)
    plt.title('Movie Clusters')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

# Visualize clusters for each method
visualize_clusters(svd_sgd)
visualize_clusters(svd_sgld)
visualize_clusters(svd_sghmc)

class HybridRecommender:
    def __init__(self, svd_model, k=10):
        self.svd_model = svd_model
        self.k = k

    def fit(self, train_data):
        self.svd_model.fit(train_data)
        self.movie_factors = self.svd_model.item_factors

    def recommend(self, user, movie_ids):
        knn = NearestNeighbors(n_neighbors=self.k, metric='cosine')
        knn.fit(self.movie_factors)
        movie_indices = [self.svd_model.item_map.get(movie_id) for movie_id in movie_ids if movie_id in self.svd_model.item_map]
        if not movie_indices:
            return []
        distances, knn_indices = knn.kneighbors(self.movie_factors[movie_indices], n_neighbors=self.k)
        recommendations = []
        for i in range(len(movie_indices)):
            svd_scores = [self.svd_model.predict(user, list(self.svd_model.item_map.keys())[idx]) for idx in knn_indices[i]]
            recommendations.extend(zip([list(self.svd_model.item_map.keys())[idx] for idx in knn_indices[i]], svd_scores))
        recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:10]  # Take top 10 recommendations
        return recommendations

# Use the HybridRecommender with the SGD optimizer to make recommendations and visualize the result
hybrid_recommender = HybridRecommender(svd_sgd)
hybrid_recommender.fit(train_data)

# Example recommendation for a user (e.g., user with ID 1)
user_id = 1
movie_ids = test_data['movieId'].unique()[:100]  # Use a subset of movie IDs for demonstration
recommendations = hybrid_recommender.recommend(user_id, movie_ids)

print("Top 10 movie recommendations for user ID 1:")
for movie_id, score in recommendations[:10]:
    print(f"Movie ID: {movie_id}, Predicted Rating: {score}")

# Visualize the recommendations
def visualize_recommendations(recommendations):
    movie_ids, scores = zip(*recommendations)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(movie_ids)), scores, alpha=0.7)
    plt.xticks(range(len(movie_ids)), movie_ids)
    plt.xlabel('Movie ID')
    plt.ylabel('Predicted Rating')
    plt.title('Top 10 Movie Recommendations')
    plt.show()

visualize_recommendations(recommendations)
