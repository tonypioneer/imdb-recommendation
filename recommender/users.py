import pandas as pd
import numpy as np
from surprise import Reader, Dataset
from surprise import KNNBaseline, KNNWithMeans, KNNBasic
from constants import DATA_PATHS
from utils.load_data import df_movie, df_train, df_test

class user_knn:
    def __init__(self, mode=0):
        self.reader = Reader()
        data = Dataset.load_from_df(
            df_train[['userId', 'movieId', 'rating']],
            self.reader
        )
        trainset = data.build_full_trainset()
        sim_options = {'name': 'pearson_baseline', 'user_based': True}
        if mode == 0:
            self.algo = KNNBaseline(sim_options=sim_options)
        elif mode == 1:
            self.algo = KNNWithMeans(sim_options=sim_options)
        elif mode == 2:
            self.algo = KNNBasic(sim_options=sim_options)
        else:
            exit(0)
        self.userid = df_test['userId'].unique().tolist()
        self.algo.fit(trainset)

    def get_similar_users(self, usrID, num=10):
        user_inner_id = self.algo.trainset.to_inner_uid(usrID)
        user_neighbors = self.algo.get_neighbors(user_inner_id, k=num)
        user_neighbors = [self.algo.trainset.to_raw_uid(inner_id)
                          for inner_id in user_neighbors]
        return user_neighbors

    def debug(self):
        similar_users = self.get_similar_users(1, 1)
        print(df_train[df_train.userId == 1].head())
        for i in similar_users:
            print(list(df_train[df_train.userId == i]['movieId']))

    def recommend(self, usrID, num=5):
        existed_movie = list(
            df_train[df_train.userId == usrID]['movieId']
        )
        similar_users = self.get_similar_users(usrID, num)
        movies_dict = {}
        for i in similar_users:
            movie = list(df_train[df_train.userId == i]['movieId'])
            vote = list(df_train[df_train.userId == i]['rating'])
            for j in range(len(vote)):
                if not (movie[j] in existed_movie):
                    if movie[j] in movies_dict.keys():
                        movies_dict[movie[j]] += vote[j]
                    else:
                        movies_dict[movie[j]] = vote[j]
        result = sorted(movies_dict.items(), key=lambda x: x[1], reverse=True)
        result = result[:num]
        recommending = []
        recommending_id = []
        for i in result:
            recommending.append(df_movie[df_movie.movieId == i[0]]['title'])
            recommending_id.append(i[0])
        return recommending, recommending_id

    def test(self, num=10):
        results = []
        for user in self.userid:
            _, ids = self.recommend(user, num)
            results.append(ids)

        df_results = pd.DataFrame(results)
        df_results['result'] = df_results['result'].apply(
            lambda x: ','.join(map(str, x)))
        df_results.to_csv(DATA_PATHS.RESULT_DATASET, index=False)

    def get_user_embeddings_knn(self):
        # Extract the user embeddings from the similarity matrix
        num_users = len(self.algo.trainset.all_users())
        user_embeddings = np.zeros((num_users, num_users))
        for user_id in range(num_users):
            user_embeddings[user_id] = self.algo.sim[user_id]
        return user_embeddings
