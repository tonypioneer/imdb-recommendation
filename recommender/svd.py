import pandas as pd
import numpy as np
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
from utils.load_data import df, df_test

class pure_svd:
    def __init__(self, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02):
        self.index = pd.read_csv('data/input/movies.csv')
        self.reader = Reader()
        self.ratings = pd.read_csv('data/input/ratings.csv')
        data = Dataset.load_from_df(self.ratings[['userId', 'movieId', 'rating']], self.reader)
        trainset = data.build_full_trainset()
        
        self.algo = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
        
        results = cross_validate(
            self.algo,
            data,
            measures=['RMSE', 'MAE'],
            cv=5,
            verbose=True
        )
        print(results)
        
        self.algo.fit(trainset)
        
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
        print(self.index[self.index.movieId == movieID]['title'])
        print('\n\n')
        
    def recommend(self, usrID, num=10):
        self.showSeenMovies(usrID)
        existed_movie = list(self.ratings[self.ratings.userId == usrID]['movieId'])
        movie_scores = {}
        
        for movieID in self.index['movieId']:
            if movieID not in existed_movie:
                est = self.algo.predict(usrID, movieID).est
                movie_scores[movieID] = est
                
        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        top_movies = sorted_movies[:num]
        
        recommending = []
        for movieID, score in top_movies:
            recommending.append(self.index[self.index.movieId == movieID]['title'])
        
        return recommending

    def get_movie_embeddings_svd(self):
        # Extract movie embeddings from the SVD model
        movie_embeddings = self.algo.qi
        return movie_embeddings

if __name__ == '__main__':
    test = pure_svd()
    result = test.recommend(34, 10)
    for i in result:
        print(i.values[0])
