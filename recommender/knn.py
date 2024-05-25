import pandas as pd
import numpy as np
import os
from surprise import Reader, Dataset, SVD
from surprise import KNNBaseline
from surprise import KNNWithMeans
from surprise import KNNBasic
from surprise.model_selection import cross_validate
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import sys
import os
sys.path.insert(0, '../docs')
from recommender.users import user_knn
from recommender.movies import movie_svd, movie_knn

# First, use KNN to match the similarity of the input user, and then select the
# 10 closest other users. Then for the selected movies, recommend the top ten
# movies based on the similarity with the movie given by the user. The input is
# the user ID and the movie ID

class knn_all:
    def __init__(self, mode=0):
        # self.movie = Movie_KNN_recommender()
        self.user = user_knn()
        self.index = pd.read_csv('data/input/movies.csv')
        self.reader = Reader()
        self.ratings = pd.read_csv('data/input/ratings.csv')
        data = Dataset.load_from_df(self.ratings[['userId', 'movieId', 'rating']], self.reader)
        trainset = data.build_full_trainset()
        sim_options = {'name': 'pearson_baseline', 'user_based': False}
        if mode == 0:
            self.algo = KNNBaseline(sim_options=sim_options)
        elif mode == 1:
            self.algo = KNNWithMeans(sim_options=sim_options)
        elif mode == 2:
            self.algo = KNNBasic(sim_options=sim_options)
        else:
            exit(0)

        results = cross_validate(
            self.algo,
            data,
            measures=['RMSE', 'MAE'],
            cv=5,
            verbose=True
        )
        print(results)
        self.algo.fit(trainset)
        self.sim = self.algo.compute_similarities()
    def cal_similarity(self, movieID, waitingID):
        movie_inner_id = self.algo.trainset.to_inner_iid(movieID)
        waiting_inner_id = self.algo.trainset.to_inner_iid(waitingID)
        return self.sim[movie_inner_id, waiting_inner_id]
    def showSeenMovies(self, usrID):
        print("\n\nThe user has seen movies below: ")
        movies = []
        for i in range(len(self.ratings['userId'])):
            if self.ratings['userId'][i] == usrID:
                movies.append(self.index[self.index.movieId == self.ratings['movieId'][i]]['title'])
        for i in movies:
            print(i.values[0])
    def showInputMovie(self, movieID):
        print("\n\nThe user's input movie is: ")
        print(self.index[self.index.movieId==movieID]['title'])
        print('\n\n')
    def recommend(self, usrID, movieID, num=10):
        self.showSeenMovies(usrID)
        self.showInputMovie(movieID)
        _, first_ids = self.user.recommend(usrID, 50)

        similarity = {}
        for i in first_ids:
            similarity[i] = self.cal_similarity(movieID, i)
        result = sorted(similarity.items(), key=lambda x: x[1], reverse=True)
        result = result[:num]
        movie = []
        for i in result:
            movie.append(self.index[self.index.movieId == i[0]]['title'])
        return movie


if __name__ == '__main__':

    test = knn_all()
    result = test.recommend(34,480)

    for i in result:
        print(i.values[0])
    