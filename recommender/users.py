# import csv
# import pandas as pd
#
# from surprise import Reader, Dataset
# from surprise import KNNBaseline
# from surprise import KNNWithMeans
# from surprise import KNNBasic
#
# from utils.load_data import df, df_movie, df_rating, df_train, df_test
#
#
# class Personal_KNN_recommender:
#     def __init__(self, mode=0):
#         self.index = df_movie
#         self.reader = Reader()
#         self.ratings = df
#         self.testings = pd.read_csv('data/output/test.csv')
#         data = Dataset.load_from_df(self.ratings[['userId', 'movieId', 'rating']], self.reader)
#         trainset = data.build_full_trainset()
#         sim_options = {'name': 'pearson_baseline', 'user_based': True}
#         if mode == 0:
#             self.algo = KNNBaseline(sim_options=sim_options)
#         elif mode == 1:
#             self.algo = KNNWithMeans(sim_options=sim_options)
#         elif mode == 2:
#             self.algo = KNNBasic(sim_options=sim_options)
#         else:
#             exit(0)
#         self.userid = []
#         for i in range(len(self.testings['userId'])):
#             if not self.testings['userId'][i] in self.userid:
#                 self.userid.append(self.testings['userId'][i])
#         self.algo.fit(trainset)
#
#     def get_similar_users(self, usrID, num=10):
#         user_inner_id = self.algo.trainset.to_inner_uid(usrID)
#         user_neighbors = self.algo.get_neighbors(user_inner_id, k=num)
#         user_neighbors = [self.algo.trainset.to_raw_uid(inner_id) for inner_id in user_neighbors]
#         # print(user_neighbors)
#         return user_neighbors
#
#     def debug(self):
#         similar_users = self.get_similar_users(1, 1)
#         print(self.ratings[self.ratings.userId == 1].head())
#         for i in similar_users:
#             print(list(self.ratings[self.ratings.userId == i]['movieId']))
#
#     def recommend(self, usrID, num=5):
#         existed_movie = list(self.ratings[self.ratings.userId==usrID]['movieId'])
#         similar_users = self.get_similar_users(usrID, num)
#         movies_dict = {}
#         for i in similar_users:
#             movie = list(self.ratings[self.ratings.userId == i]['movieId'])
#             vote = list(self.ratings[self.ratings.userId == i]['rating'])
#             for j in range(len(vote)):
#                 if not (movie[j] in existed_movie):
#                     if movie[j] in movies_dict.keys():
#                         movies_dict[movie[j]] += vote[j]
#                     else:
#                         movies_dict[movie[j]] = vote[j]   # 从最相似的用户中挑选出没看过的电影，评分相加
#         result = sorted(movies_dict.items(), key=lambda x: x[1], reverse=True)  # 对评分进行排序
#         result = result[:num]  # 挑选出最高评分的10部电影
#         # print(result)
#         recommending = []
#         recommending_id = []
#         for i in result:
#             recommending.append(self.index[self.index.movieId==i[0]]['title'])
#             recommending_id.append(i[0])
#         return recommending, recommending_id  # 返回推荐的电影名字和id
#
#     def test(self, num = 10):
#         result = []
#         for user in self.userid:
#             _, ids = self.recommend(user, num)
#             # print(ids)
#             result.append(ids)
#
#         with open("result.csv", "w") as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow(['userId', 'result'])
#             for i,row in enumerate(result):
#                 writer.writerow([self.userid[i], row])

import csv
import pandas as pd
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

        with open("result.csv", "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['userId', 'result'])
            for user, movie_ids in results:
                writer.writerow([user, movie_ids])
