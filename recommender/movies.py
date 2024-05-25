from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
from utils.load_data import df_rating, df_movie
import numpy as np

class movie_svd:
    def __init__(self):
        self.reader = Reader()
        data = Dataset.load_from_df(
            df_rating[['userId', 'movieId', 'rating']],
            self.reader
        )
        self.n_factors = 100  # Number of factors used in SVD
        self.svd = SVD(n_epochs=20, n_factors=self.n_factors, verbose=True)
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
        result = sorted(dic.items(), key=lambda x: x[1], reverse=True)
        result = result[:num]
        movie = []
        rates = []
        ids = []
        for i in result:
            movie.append(df_movie[df_movie.movieId == i[0]]['title'])
            rates.append(i[1])
            ids.append(i[0])
        return movie, ids

    def get_movie_embeddings_svd(self):
        # Get the number of movies and factors
        num_movies = len(df_rating['movieId'].unique())
        movie_embeddings = np.zeros((num_movies, self.n_factors))
        for movie_id in range(num_movies):
            movie_inner_id = self.svd.trainset.to_inner_iid(movie_id)
            movie_embeddings[movie_id] = self.svd.qi[movie_inner_id]
        return movie_embeddings
    
if __name__ == '__main__':

    test = movie_svd()
    result = test.recommend(122922, 10)
    for i in result:
        print(i.values[0])



