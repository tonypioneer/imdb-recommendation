import numpy as np
import pandas as pd

from utils.load_data import df
from recommender.users import Personal_KNN_recommender
from utils.dataset import make_training_set


if __name__ == '__main__':
    # Create a matrix of ratings with users as rows and movies as columns

    # Find the maximum user id and movie id
    max_user_id = df['userId'].max()
    max_movie_id = df['movieId'].max()

    # Create a zero matrix of size max_user_id x max_movie_id
    rating_matrix = np.zeros((max_user_id, max_movie_id))

    # Iterate over each row of df
    for index, row in df.iterrows():
        user_index = row['userId'] - 1
        movie_index = row['movieId'] - 1
        rating = row['rating']

        # Update the value at the corresponding position in rating_matrix
        # to rating
        rating_matrix[user_index, movie_index] = rating

    # test = Personal_KNN_recommender()
    # result = test.recommend(6, 10)
    # for i in result:
    #     print(i)
    #
    # test.test(10)

    make_training_set()
