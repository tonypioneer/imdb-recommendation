import pandas as pd

from constants import DATA_PATHS
from utils.dataset import make_training_set


def load_data():
    """Load the data of movies and ratings from the csv files.

    :return:
        df (pd.DataFrame): the merged dataframe of movies and ratings
        df_movie (pd.DataFrame): the dataframe of movies
        df_rating (pd.DataFrame): the dataframe of ratings
    """
    # Load rating data
    df_rating = pd.read_csv(
        DATA_PATHS.RATING_DATASET,
        sep=',',
        low_memory=False
    )

    # Load movie data
    df_movie = pd.read_csv(
        DATA_PATHS.MOVIE_DATASET,
        sep=',',
        low_memory=False
    )

    # Merge the dataframes on the movieId column
    df = pd.merge(df_rating, df_movie, on=['movieId'])

    # Generate training set and test set
    df_train, df_test = make_training_set(df_rating)

    return df, df_movie, df_rating, df_train, df_test


df, df_movie, df_rating, df_train, df_test = load_data()
