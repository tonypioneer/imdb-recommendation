import numpy as np

from utils.load_data import df

import pandas as pd
import sys
import os
import csv

sys.path.insert(0, 'docs')
from recommender.movies import Movie_KNN_recommender
from recommender.users import Personal_KNN_recommender
from recommender.movies import Personal_SVD_recommender


# 首先用KNN对输入的用户进行相似度匹配，然后挑选出最接近的10个其他用户
# 之后对于选出的电影，根据SVD计算用户对电影的模拟评分来进行排序

class KNN_SVD_ensemble:
    def __init__(self):
        self.user = Personal_KNN_recommender()
        self.movie = Personal_SVD_recommender()
        self.testings = pd.read_csv('data/output/test.csv')
        self.userid = []
        for i in range(len(self.testings['userId'])):
            if not self.testings['userId'][i] in self.userid:
                self.userid.append(self.testings['userId'][i])

    def recommend(self, usrID):
        first_ids = self.user.recommend(usrID, 50)
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

        with open("data/output/result.csv", "w") as csvfile:
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

    test = KNN_SVD_ensemble()
    # test.recommend(2)
    test.test(10)