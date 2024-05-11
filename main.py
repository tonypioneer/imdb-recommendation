import numpy as np
import pandas as pd

from constants import DATA_PATHS

df = pd.read_csv(DATA_PATHS.RATING_DATASET, sep=',')
df_id = pd.read_csv(DATA_PATHS.MOVIE_DATASET, sep=',')
df = pd.merge(df, df_id, on=['movieId'])
rating_matrix = np.zeros((df.userId.unique().shape[0], max(df.movieId)))
for row in df.itertuples():
    rating_matrix[row[1]-1, row[2]-1] = row[3]
    rating_matrix = rating_matrix[:,:9000]
