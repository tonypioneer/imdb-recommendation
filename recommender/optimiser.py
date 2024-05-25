import numpy as np

class SVD_SGD:
    def __init__(self, n_factors, lr, reg, n_epochs):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs

    def fit(self, train_data):
        self.user_factors = np.random.normal(
            0,
            0.1,
            (train_data['userId'].max()+1, self.n_factors)
        )
        self.item_factors = np.random.normal(
            0,
            0.1,
            (train_data['movieId'].max()+1, self.n_factors)
        )
        for epoch in range(self.n_epochs):
            for row in train_data.itertuples():
                user, item, rating = row.userId, row.movieId, row.rating
                error = rating - self.predict(user, item)
                self.user_factors[user] += (
                        self.lr * (error * self.item_factors[item] -
                        self.reg * self.user_factors[user])
                )
                self.item_factors[item] += self.lr * (
                        error * self.user_factors[user] -
                        self.reg * self.item_factors[item]
                )

    def predict(self, user, item):
        return np.dot(self.user_factors[user], self.item_factors[item])
    

class SVD_SGLD:
    def __init__(self, n_factors, lr, reg, n_epochs, noise_scale):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.noise_scale = noise_scale

    def fit(self, train_data):
        self.user_factors = np.random.normal(
            0,
            0.1,
            (train_data['userId'].max()+1, self.n_factors)
        )
        self.item_factors = np.random.normal(
            0,
            0.1,
            (train_data['movieId'].max()+1, self.n_factors)
        )
        for epoch in range(self.n_epochs):
            for row in train_data.itertuples():
                user, item, rating = row.userId, row.movieId, row.rating
                error = rating - self.predict(user, item)
                noise = np.random.normal(
                    0,
                    self.noise_scale,
                    self.n_factors
                )
                self.user_factors[user] += self.lr * (
                        error * self.item_factors[item] -
                        self.reg * self.user_factors[user]
                ) + noise
                self.item_factors[item] += self.lr * (
                        error * self.user_factors[user] -
                        self.reg * self.item_factors[item]
                ) + noise

    def predict(self, user, item):
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
        self.user_factors = np.random.normal(
            0,
            0.1,
            (train_data['userId'].max()+1, self.n_factors)
        )
        self.item_factors = np.random.normal(
            0,
            0.1,
            (train_data['movieId'].max()+1, self.n_factors)
        )
        for epoch in range(self.n_epochs):
            for row in train_data.itertuples():
                user, item, rating = row.userId, row.movieId, row.rating
                error = rating - self.predict(user, item)
                noise = np.random.normal(
                    0,
                    self.noise_scale, self.n_factors
                )
                self.user_factors[user] += self.lr * (
                        error * self.item_factors[item] -
                        self.reg * self.user_factors[user]
                ) - self.friction * self.user_factors[user] + noise
                self.item_factors[item] += self.lr * (
                        error * self.user_factors[user] -
                        self.reg * self.item_factors[item]
                ) - self.friction * self.item_factors[item] + noise

    def predict(self, user, item):
        return np.dot(self.user_factors[user], self.item_factors[item])
    
from sklearn.neighbors import NearestNeighbors

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
        movie_indices = [self.svd_model.item_map.get(movie_id)
                         for movie_id in movie_ids
                         if movie_id in self.svd_model.item_map]
        if not movie_indices:
            return []
        distances, knn_indices = knn.kneighbors(
            self.movie_factors[movie_indices], n_neighbors=self.k
        )
        recommendations = []
        for i in range(len(movie_indices)):
            svd_scores = [self.svd_model.predict(
                user,
                list(self.svd_model.item_map.keys())[idx]
            ) for idx in knn_indices[i]]
            recommendations.extend(zip(
                [list(self.svd_model.item_map.keys())[idx]
                 for idx in knn_indices[i]], svd_scores
            ))
        recommendations = sorted(
            recommendations,
            key=lambda x: x[1],
            reverse=True
        )[:10]  # Take top 10 recommendations
        return recommendations
