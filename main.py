import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils.load_data import df, df_test
from recommender.users import user_knn
from recommender.movies import movie_svd
from constants import DATA_PATHS
from recommender.knn import knn_all
import time

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

    print("Reducing embeddings for hybrid method")
    pca = PCA(n_components=2)
    reduced_user_data_hybrid = pca.fit_transform(user_embeddings_hybrid)
    reduced_movie_data_hybrid = pca.fit_transform(movie_embeddings_hybrid)

    print("Reducing embeddings for KNN method")
    reduced_user_data_knn = pca.fit_transform(user_embeddings_knn)
    reduced_movie_data_knn = pca.fit_transform(movie_embeddings_knn)

    subset_user_ids = df['userId'].unique()[:len(reduced_user_data_hybrid)]
    subset_movie_ids = df['movieId'].unique()[:len(reduced_movie_data_hybrid)]

    print("Plotting clusters for Hybrid method (User Clusters)")
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(reduced_user_data_hybrid[:, 0], reduced_user_data_hybrid[:, 1], c=subset_user_ids, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('User Clusters (Hybrid Method)')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

    print("Plotting clusters for Hybrid method (Movie Clusters)")
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(reduced_movie_data_hybrid[:, 0], reduced_movie_data_hybrid[:, 1], c=subset_movie_ids, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('Movie Clusters (Hybrid Method)')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

    print("Plotting clusters for KNN method (User Clusters)")
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(reduced_user_data_knn[:, 0], reduced_user_data_knn[:, 1], c=subset_user_ids, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('User Clusters (KNN Method)')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

    print("Plotting clusters for KNN method (Movie Clusters)")
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(reduced_movie_data_knn[:, 0], reduced_movie_data_knn[:, 1], c=subset_movie_ids, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('Movie Clusters (KNN Method)')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()
