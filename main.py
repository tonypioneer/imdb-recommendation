# import numpy as np
# import pandas as pd

from utils.load_data import df
from recommender.users import Personal_KNN_recommender


if __name__ == '__main__':
    test = Personal_KNN_recommender()
    result = test.recommend(6, 10)
    for i in result:
        print(i)

    test.test(10)
