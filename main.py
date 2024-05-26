import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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

    # test = movie_recommender()
    # test.recommend(2)
    # test.test()

    # test = knn_all()
    # result = test.recommend(34, 480)
    #
    # for i in result:
    #     print(i.values[0])

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
    avg_ratings_array = np.array([avg_ratings.get(mid, 0) for mid in df_movie['movieId']])

    # inertia = []
    # K_range = range(10, 150, 10)
    # for K in K_range:
    #     kmeans = KMeans(n_clusters=K, random_state=0).fit(user_embeddings_hybrid)
    #     inertia.append(kmeans.inertia_) # Plot the elbow graph
    # plt.figure(figsize=(8, 6))
    # plt.plot(K_range, inertia, 'bo-')
    # plt.xlabel('Number of clusters (K)')
    # plt.ylabel('Inertia')
    # plt.title('Elbow Method For Optimal K (user_embeddings_hybrid)')
    # plt.show()
    #
    # inertia = []
    # K_range = range(10, 150, 10)
    # for K in K_range:
    #     kmeans = KMeans(n_clusters=K, random_state=0).fit(movie_embeddings_hybrid)
    #     inertia.append(kmeans.inertia_) # Plot the elbow graph
    # plt.figure(figsize=(8, 6))
    # plt.plot(K_range, inertia, 'bo-')
    # plt.xlabel('Number of clusters (K)')
    # plt.ylabel('Inertia')
    # plt.title('Elbow Method For Optimal K (movie_embeddings_hybrid)')
    # plt.show()
    #
    # inertia = []
    # K_range = range(10, 150, 10)
    # for K in K_range:
    #     kmeans = KMeans(n_clusters=K, random_state=0).fit(user_embeddings_knn)
    #     inertia.append(kmeans.inertia_) # Plot the elbow graph
    # plt.figure(figsize=(8, 6))
    # plt.plot(K_range, inertia, 'bo-')
    # plt.xlabel('Number of clusters (K)')
    # plt.ylabel('Inertia')
    # plt.title('Elbow Method For Optimal K (user_embeddings_knn)')
    # plt.show()
    #
    # inertia = []
    # K_range = range(10, 150, 10)
    # for K in K_range:
    #     kmeans = KMeans(n_clusters=K, random_state=0).fit(movie_embeddings_knn)
    #     inertia.append(kmeans.inertia_) # Plot the elbow graph
    # plt.figure(figsize=(8, 6))
    # plt.plot(K_range, inertia, 'bo-')
    # plt.xlabel('Number of clusters (K)')
    # plt.ylabel('Inertia')
    # plt.title('Elbow Method For Optimal K (movie_embeddings_knn)')
    # plt.show()

    # Reduce embeddings to 3D for visualization
    print("Reducing embeddings for hybrid method to 3D")
    pca_3d = PCA(n_components=3)
    reduced_user_data_hybrid_3d = pca_3d.fit_transform(user_embeddings_hybrid)
    reduced_movie_data_hybrid_3d = pca_3d.fit_transform(movie_embeddings_hybrid)

    print("Reducing embeddings for KNN method to 3D")
    reduced_user_data_knn_3d = pca_3d.fit_transform(user_embeddings_knn)
    reduced_movie_data_knn_3d = pca_3d.fit_transform(movie_embeddings_knn)

    # Apply K-means clustering
    kmeans_user_hybrid = KMeans(n_clusters=3).fit(reduced_user_data_hybrid_3d)
    kmeans_movie_hybrid = KMeans(n_clusters=3).fit(reduced_movie_data_hybrid_3d)
    kmeans_user_knn = KMeans(n_clusters=3).fit(reduced_user_data_knn_3d)
    kmeans_movie_knn = KMeans(n_clusters=3).fit(reduced_movie_data_knn_3d)

    # Select a subset of user IDs and movie IDs to match the reduced data points
    subset_user_ids = df['userId'].unique()[:len(reduced_user_data_hybrid_3d)]
    subset_movie_ids = df['movieId'].unique()[:len(reduced_movie_data_hybrid_3d)]

    random_user_indices = np.random.choice(user_embeddings_hybrid.shape[0], 100, replace=False)
    subset_user_ids = df['userId'].unique()[random_user_indices]
    
    # 3D Scatter Plot for User Clusters (Hybrid Method)
    print("Plotting 3D clusters for Hybrid method (User Clusters)")
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(reduced_user_data_hybrid_3d[:, 0], reduced_user_data_hybrid_3d[:, 1], reduced_user_data_hybrid_3d[:, 2], c=kmeans_user_hybrid.labels_, cmap='viridis', alpha=0.6)
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
    scatter = ax.scatter(reduced_movie_data_hybrid_3d[:, 0], reduced_movie_data_hybrid_3d[:, 1], reduced_movie_data_hybrid_3d[:, 2], c=kmeans_movie_hybrid.labels_, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Average Rating')
    ax.set_title('Movie Clusters (Hybrid Method)')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    plt.show()

    # 3D Scatter Plot for User Clusters (KNN Method)
    print("Plotting 3D clusters for KNN method (User Clusters)")
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(reduced_user_data_knn_3d[:, 0], reduced_user_data_knn_3d[:, 1], reduced_user_data_knn_3d[:, 2], c=kmeans_user_knn.labels_, cmap='viridis', alpha=0.6)
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
    scatter = ax.scatter(reduced_movie_data_knn_3d[:, 0], reduced_movie_data_knn_3d[:, 1], reduced_movie_data_knn_3d[:, 2], c=kmeans_movie_knn.labels_, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Average Rating')
    ax.set_title('Movie Clusters (KNN Method)')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    plt.show()

    # 2D Scatter Plot for User Clusters (Hybrid Method)
    print("Plotting 2D clusters for Hybrid method (User Clusters)")
    pca_2d = PCA(n_components=2)
    reduced_user_data_hybrid_2d = pca_2d.fit_transform(user_embeddings_hybrid)
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(reduced_user_data_hybrid_2d[:, 0], reduced_user_data_hybrid_2d[:, 1], c=kmeans_user_hybrid.labels_, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('User Clusters (Hybrid Method)')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

    # 2D Scatter Plot for Movie Clusters (Hybrid Method)
    print("Plotting 2D clusters for Hybrid method (Movie Clusters)")
    reduced_movie_data_hybrid_2d = pca_2d.fit_transform(movie_embeddings_hybrid)
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(reduced_movie_data_hybrid_2d[:, 0], reduced_movie_data_hybrid_2d[:, 1], c=kmeans_movie_hybrid.labels_, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Average Rating')
    plt.title('Movie Clusters (Hybrid Method)')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

    # 2D Scatter Plot for User Clusters (KNN Method)
    print("Plotting 2D clusters for KNN method (User Clusters)")
    reduced_user_data_knn_2d = pca_2d.fit_transform(user_embeddings_knn)
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(reduced_user_data_knn_2d[:, 0], reduced_user_data_knn_2d[:, 1], c=kmeans_user_knn.labels_, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('User Clusters (KNN Method)')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

    # 2D Scatter Plot for Movie Clusters (KNN Method)
    print("Plotting 2D clusters for KNN method (Movie Clusters)")
    reduced_movie_data_knn_2d = pca_2d.fit_transform(movie_embeddings_knn)
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(reduced_movie_data_knn_2d[:, 0], reduced_movie_data_knn_2d[:, 1], c=kmeans_movie_knn.labels_, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Average Rating')
    plt.title('Movie Clusters (KNN Method)')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

    # 2D Scatter Plot with Movie Titles
    print("Plotting 2D scatter plot with movie titles")
    movie_titles = df_movie['title'].unique()[:30]  # Limit to 30 movie titles
    reduced_movie_data_titles = PCA(n_components=2).fit_transform(movie_embeddings_knn[:30])  # Plot only the first 30 movies

    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(reduced_movie_data_titles[:, 0], reduced_movie_data_titles[:, 1], alpha=0.6, c=avg_ratings_array[:30], cmap='viridis')
    plt.colorbar(scatter, label='Average Rating')
    for i, title in enumerate(movie_titles):
        ax.text(reduced_movie_data_titles[i, 0], reduced_movie_data_titles[i, 1], title, fontsize=9)
    ax.set_title('Movie Titles in 2D PCA Space')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    plt.show()
