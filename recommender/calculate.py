import pandas as pd
from utils.load_data import df_test
from constants import DATA_PATHS


def calculate():
    # Load data
    usrid = df_test['userId'].unique().tolist()
    movieid = df_test['movieId'].unique().tolist()

    data_all = []
    index = 0
    for user in usrid:
        this_user = []
        if index >= len(df_test['userId']):
            break
        while df_test['userId'][index] == user:
            index += 1
            if index >= len(df_test['userId']):
                break
            this_user.append(df_test['movieId'][index])
        print(len(this_user))
        data_all.append(this_user)

    print('data all', len(data_all))

    result = pd.read_csv(DATA_PATHS.RESULT_DATASET)
    print('pred', len(result['userId']))
    posi = 0
    neg = 0
    for i in range(len(result['userId'])):
        print(i)
        temp = result['result'][i]
        temp = temp[1:-1].split(',')
        temp = [int(x) for x in temp]
        # print(temp)
        # break
        # print(temp)
        # print(data_all[i])
        # break
        for movieid in list(temp):
            if movieid in data_all[i]:
                posi += 1
            else:
                neg += 1

    print(posi, neg, posi / float(posi + neg))
