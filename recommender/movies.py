import pandas as pd
import numpy as np
import os
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

class Personal_SVD_recommender:
    def __init__(self):
        self.reader = Reader()
        self.ratings = pd.read_csv('data/output/train.csv')
        data = Dataset.load_from_df(self.ratings[['userId', 'movieId', 'rating']], self.reader)
        # data.split(n_folds=5)
        self.svd = SVD(n_epochs=20, n_factors=100, verbose=True)
        results = cross_validate(self.svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
        trainset = data.build_full_trainset()
        self.svd.fit(trainset)

        self.index = pd.read_csv('data/input/movies.csv')

    def rating(self, usrID, movieID):
        rate = self.svd.predict(usrID, movieID)
        return rate[3]
    def sample_movies(self, n):
        pass

    # 主要用于对给定的列表，进行用户模拟评分，之后进一步挑选，属于二阶步骤
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
            movie.append(self.index[self.index.movieId==i[0]]['title'])
            rates.append(i[1])
            ids.append(i[0])
        return movie, ids
