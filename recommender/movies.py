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
        trainset = data.build_full_trainset()
        self.svd.fit(trainset)

    def rating(self, usrID, movieID):
        rate = self.svd.predict(usrID, movieID)
        return rate[3]
<<<<<<< Updated upstream
=======
    
    def sample_movies(self, n):
        pass
>>>>>>> Stashed changes

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

