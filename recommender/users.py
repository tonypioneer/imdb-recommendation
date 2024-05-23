import csv
from surprise import Reader, Dataset
from surprise import KNNBaseline, KNNWithMeans, KNNBasic
from utils.load_data import df_movie, df_rating, df_train, df_test


class Personal_KNN_recommender:
    def __init__(self, mode=0):
        self.index = df_movie
        self.reader = Reader()
        self.ratings = df_rating
        self.testings = df_test
        data = Dataset.load_from_df(
            self.ratings[['userId', 'movieId', 'rating']], self.reader)
        trainset = data.build_full_trainset()
        sim_options = {'name': 'pearson_baseline', 'user_based': True}

        if mode == 0:
            self.algo = KNNBaseline(sim_options=sim_options)
        elif mode == 1:
            self.algo = KNNWithMeans(sim_options=sim_options)
        elif mode == 2:
            self.algo = KNNBasic(sim_options=sim_options)
        else:
            raise ValueError("Invalid mode selected. Choose 0, 1, or 2.")

        self.algo.fit(trainset)
        self.userid = self.testings['userId'].unique().tolist()

    def get_similar_users(self, usrID, num=10):
        user_inner_id = self.algo.trainset.to_inner_uid(usrID)
        user_neighbors = self.algo.get_neighbors(user_inner_id, k=num)
        return [self.algo.trainset.to_raw_uid(inner_id) for inner_id in
                user_neighbors]

    def recommend(self, usrID, num=5):
        existed_movies = set(
            self.ratings[self.ratings.userId == usrID]['movieId'])
        similar_users = self.get_similar_users(usrID, num)
        movies_dict = {}

        for user in similar_users:
            user_ratings = self.ratings[self.ratings.userId == user]
            for _, row in user_ratings.iterrows():
                movie_id, rating = row['movieId'], row['rating']
                if movie_id not in existed_movies:
                    if movie_id in movies_dict:
                        movies_dict[movie_id] += rating
                    else:
                        movies_dict[movie_id] = rating

        top_movies = sorted(movies_dict.items(), key=lambda x: x[1],
                            reverse=True)[:num]
        return [(self.index[self.index.movieId == movie_id]['title'].iloc[0],
                 movie_id) for movie_id, _ in top_movies]

    def test(self, num=10):
        results = [(user, self.recommend(user, num)[1]) for user in self.userid]

        with open("data/output/result.csv", "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['userId', 'result'])
            for user, movie_ids in results:
                writer.writerow([user, movie_ids])
