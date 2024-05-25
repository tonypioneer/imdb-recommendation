from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
from utils.load_data import df_rating, df_movie

class movie_svd:
    def __init__(self):
        self.reader = Reader()
        data = Dataset.load_from_df(
            df_rating[['userId', 'movieId', 'rating']],
            self.reader
        )
        # data.split(n_folds=5)
        self.svd = SVD(n_epochs=20, n_factors=100, verbose=True)
        results = cross_validate(
            self.svd,
            data,
            measures=['RMSE', 'MAE'],
            cv=5,
            verbose=True
        )
        print(results)
        trainset = data.build_full_trainset()
        self.svd.fit(trainset)

    def rating(self, usrID, movieID):
        rate = self.svd.predict(usrID, movieID)
        return rate[3]

    # Used to simulate user ratings for a given list, and then further select,
    # which is a second-order step
    def recommend(self, usrID, movies, num=10):
        dic = {}
        for i in movies:
            dic[i] = self.rating(usrID, i)
        # print('dic', dic)
        result = sorted(dic.items(), key=lambda x: x[1], reverse=True)
        result = result[:num]
        # print('result', result)
        movie = []
        rates = []
        ids = []
        for i in result:
            # print(i)
            movie.append(df_movie[df_movie.movieId==i[0]]['title'])
            rates.append(i[1])
            ids.append(i[0])
        return movie, ids
    

import pandas as pd
import numpy as np
import os
from surprise import Reader, Dataset, SVD
from surprise import KNNBaseline
from surprise import KNNWithMeans
from surprise import KNNBasic


class movie_knn:
    def __init__(self, mode=0):
        self.index = pd.read_csv('data/personal/movies.csv')
        self.reader = Reader()
        self.ratings = pd.read_csv('data/personal/ratings.csv')
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

        self.algo.fit(trainset)

    def get_similar_movies(self, movieID, num=10):
        movie_inner_id = self.algo.trainset.to_inner_iid(movieID)
        movie_neighbors = self.algo.get_neighbors(movie_inner_id, k=num)
        movie_neighbors = [self.algo.trainset.to_raw_iid(inner_id) for inner_id in movie_neighbors]
        print(movie_neighbors)
        return movie_neighbors

    def debug(self):
        similar_users = self.get_similar_movies(1, 1)
        print(self.ratings[self.ratings.userId == 1].head())
        for i in similar_users:
            print(list(self.ratings[self.ratings.userId == i]['movieId']))

    def recommend(self, movieID, num=10):
        movie_similar = self.get_similar_movies(movieID, num)
        recommending = []
        for i in movie_similar:
            recommending.append(self.index[self.index.movieId == i]['title'])
        return recommending


if __name__ == '__main__':

    test = movie_knn()
    result = test.recommend(122922, 10)
    for i in result:
        print(i.values[0])



