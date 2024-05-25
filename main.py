import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils.load_data import df, df_test, df_movie, df_rating
from recommender.users import user_knn
from recommender.movies import movie_svd
from constants import DATA_PATHS
from recommender.knn import knn_all
import time
from mpl_toolkits.mplot3d import Axes3D

class movie_recommender:
    def __init__(self):
        self.user = user_knn()
        self.movie = movie_svd()
        self.userid = df_test['userId'].unique().tolist()[:10]  # Run for only 10 users for testing

    def recommend(self, userID, first_num=50, second_num=10):
        start_time = time.time()
        _, first_ids = self.user.recommend(userID, first_num)
        print(f"User KNN recommendation for user {userID} took {time.time() - start_time} seconds")

        start_time = time.time()
        second_ids, movie_id = self.movie.recommend(userID, first_ids, second_num)
        print(f"Movie SVD recommendation for user {userID} took {time.time() - start_time} seconds")
        
        return movie_id

    def test(self):
        results = []
        for user in self.userid:
            print(f"Processing recommendations for user {user}")
            ids = self.recommend(user)
            results.append({'userId': user, 'result': ids})

        df_result = pd.DataFrame(results)
        df_result.to_csv(DATA_PATHS.RESULT_DATASET, index=False)
        
    def get_user_embeddings_hybrid(self):
        return self.user.get_user_embeddings_knn()

    def get_movie_embeddings_hybrid(self):
        movie_knn_instance = knn_all()
        return movie_knn_instance.get_movie_embeddings_knn()

if __name__ == '__main__':
    max_user_id = df['userId'].max()
    max_movie_id = df['movieId'].max()
    rating_matrix = np.zeros((max_user_id, max_movie_id))

    for index, row in df.iterrows():
        user_index = row['userId'] - 1
        movie_index = row['movieId'] - 1
        rating = row['rating']
        rating_matrix[user_index, movie_index] = rating

    print("Testing hybrid method (SVD + KNN)")
    test_hybrid = movie_recommender()
    test_hybrid.test()

    print("Testing pure KNN method")
    test_knn = knn_all()

    print("Getting embeddings for hybrid method")
    user_embeddings_hybrid = test_hybrid.get_user_embeddings_hybrid()
    movie_embeddings_hybrid = test_hybrid.get_movie_embeddings_hybrid()

    print("Getting embeddings for KNN method")
    user_embeddings_knn = test_hybrid.user.get_user_embeddings_knn()
    movie_embeddings_knn = test_knn.get_movie_embeddings_knn()

    # Calculate average ratings for movies
    avg_ratings = df_rating.groupby('movieId')['rating'].mean()
    movie_ids = df_movie['movieId'].unique()
    avg_ratings_array = np.array([avg_ratings.get(mid, 0) for mid in movie_ids])

    # Reduce embeddings to 3D for visualization
    print("Reducing embeddings for hybrid method to 3D")
    pca_3d = PCA(n_components=3)
    reduced_user_data_hybrid_3d = pca_3d.fit_transform(user_embeddings_hybrid)
    reduced_movie_data_hybrid_3d = pca_3d.fit_transform(movie_embeddings_hybrid)

    print("Reducing embeddings for KNN method to 3D")
    reduced_user_data_knn_3d = pca_3d.fit_transform(user_embeddings_knn)
    reduced_movie_data_knn_3d = pca_3d.fit_transform(movie_embeddings_knn)

    # Select a subset of user IDs and movie IDs to match the reduced data points
    subset_user_ids = df['userId'].unique()[:len(reduced_user_data_hybrid_3d)]
    subset_movie_ids = df['movieId'].unique()[:len(reduced_movie_data_hybrid_3d)]

    # 3D Scatter Plot for User Clusters (Hybrid Method)
    print("Plotting 3D clusters for Hybrid method (User Clusters)")
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(reduced_user_data_hybrid_3d[:, 0], reduced_user_data_hybrid_3d[:, 1], reduced_user_data_hybrid_3d[:, 2], c=subset_user_ids, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    ax.set_title('User Clusters (Hybrid Method)')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    plt.show()

    # 3D Scatter Plot for Movie Clusters (Hybrid Method)
    print("Plotting 3D clusters for Hybrid method (Movie Clusters)")
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(reduced_movie_data_hybrid_3d[:, 0], reduced_movie_data_hybrid_3d[:, 1], reduced_movie_data_hybrid_3d[:, 2], c=avg_ratings_array[:len(reduced_movie_data_hybrid_3d)], cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    ax.set_title('Movie Clusters (Hybrid Method)')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    plt.show()

    # 3D Scatter Plot for User Clusters (KNN Method)
    print("Plotting 3D clusters for KNN method (User Clusters)")
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(reduced_user_data_knn_3d[:, 0], reduced_user_data_knn_3d[:, 1], reduced_user_data_knn_3d[:, 2], c=subset_user_ids, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    ax.set_title('User Clusters (KNN Method)')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    plt.show()

    # 3D Scatter Plot for Movie Clusters (KNN Method)
    print("Plotting 3D clusters for KNN method (Movie Clusters)")
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(reduced_movie_data_knn_3d[:, 0], reduced_movie_data_knn_3d[:, 1], reduced_movie_data_knn_3d[:, 2], c=avg_ratings_array[:len(reduced_movie_data_knn_3d)], cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    ax.set_title('Movie Clusters (KNN Method)')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    plt.show()

    # 2D Scatter Plot with Movie Titles
    print("Plotting 2D scatter plot with movie titles")
    movie_titles = df_movie['title'].unique()[:len(reduced_movie_data_knn_3d)]
    reduced_movie_data_titles = PCA(n_components=2).fit_transform(movie_embeddings_knn)

    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(reduced_movie_data_titles[:, 0], reduced_movie_data_titles[:, 1], alpha=0.6, c=avg_ratings_array[:len(reduced_movie_data_titles)], cmap='viridis')
    plt.colorbar(scatter)
    for i, title in enumerate(movie_titles):
        ax.text(reduced_movie_data_titles[i, 0], reduced_movie_data_titles[i, 1], title, fontsize=9)
    ax.set_title('Movie Titles in 2D PCA Space')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    plt.show()
