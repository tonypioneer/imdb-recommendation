import numpy as np
import pandas as pd

from utils.load_data import df, df_test
from recommender.users import user_knn
from recommender.movies import movie_svd
from constants import DATA_PATHS
from recommender.knn import knn_all


# Use KNN to match the similarity of the input user, and then select the 10
# closest other users, then for the selected movies, sort according to the
# simulated rating of the user to the movie

class movie_recommender:
    def __init__(self):
        self.user = user_knn()
        self.movie = movie_svd()
        self.userid = df_test['userId'].unique().tolist()

    def recommend(self, userID, first_num=50, second_num=10):
        _, first_ids = self.user.recommend(userID, first_num)
        # print(first_ids)
        second_ids, movie_id = self.movie.recommend(
            userID,
            first_ids,
            second_num
        )
        # print(second_ids)
        return movie_id

    def test(self):
        results = []
        for user in self.userid:
            # print(user)
            ids = self.recommend(user)
            # print(ids)
            results.append({'userId': user, 'result': ids})

        df_result = pd.DataFrame(results)
        df_result.to_csv(DATA_PATHS.RESULT_DATASET, index=False)


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

    # test = movie_recommender()
    # # test.recommend(2)
    # test.test()

    test = knn_all()
    result = test.recommend(34, 480)

    for i in result:
        print(i.values[0])
