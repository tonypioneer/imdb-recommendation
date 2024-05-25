import numpy as np
import pandas as pd
import csv

from utils.load_data import df, df_movie, df_rating, df_train, df_test
from recommender.users import user_knn
from recommender.movies import movie_svd


# Use KNN to match the similarity of the input user, and then select the 10
# closest other users, then for the selected movies, sort according to the
# simulated rating of the user to the movie

class movie_recommender:
    def __init__(self):
        self.user = user_knn()
        self.movie = movie_svd()
        self.testings = pd.read_csv('data/output/test.csv')
        self.userid = []
        for i in range(len(self.testings['userId'])):
            if not self.testings['userId'][i] in self.userid:
                self.userid.append(self.testings['userId'][i])


    def recommend(self, usrID):
        _, first_ids = self.user.recommend(usrID, 50)
        # print(first_ids)
        second_ids, movie_id = self.movie.recommend(usrID, first_ids, 10)
        # print(second_ids)
        return movie_id

    def test(self, num):
        result = []
        for user in self.userid:
            print(user)
            ids = self.recommend(user)
            print(ids)
            result.append(ids)

        with open("result.csv", "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['userId', 'result'])
            for i, row in enumerate(result):
                writer.writerow([self.userid[i], row])


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

    test = movie_recommender()
    # test.recommend(2)
    test.test(10)
