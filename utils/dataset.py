import pandas as pd
import numpy as np
import random

from utils.load_data import df_rating
from constants import DIVISION_THRESHOLD, DATA_PATHS


def make_training_set():
    """Generate training set and test set for the user recommender.

    :return:
        df_train (pd.DataFrame): Training set.
        df_test (pd.DataFrame): Test set.
    """
    # Set random seed for reproducibility
    random.seed(42)

    # Set the column structure
    cols = ['userId', 'movieId', 'rating']

    # Create empty DataFrames to store the training and test sets
    df_train = pd.DataFrame(columns=cols)
    df_test = pd.DataFrame(columns=cols)

    # Process data by grouping by user
    grouped = df_rating.groupby('userId')

    # Traverse the data for each user
    for name, group in grouped:
        # Divide the data into training and test sets based on the threshold
        group['rand'] = np.random.rand(len(group))
        train = group[group['rand'] < DIVISION_THRESHOLD][cols]
        test = group[group['rand'] >= DIVISION_THRESHOLD][cols]

        # Add to the training and test set DataFrames
        df_train = pd.concat([df_train, train])
        df_test = pd.concat([df_test, test])

    # Output to CSV file
    df_train.to_csv(DATA_PATHS.TRAINING_DATASET, index=False)
    df_test.to_csv(DATA_PATHS.TEST_DATASET, index=False)

    return df_train, df_test
