import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Embedding, Input, Flatten, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
ratings = pd.read_csv('../data/input/ratings.csv')

# Check the maximum userId and movieId in the dataset
max_user_id = ratings['userId'].max()
max_movie_id = ratings['movieId'].max()

n_users = max_user_id + 1  # Ensure that the user IDs fit within the embedding layer
n_items = max_movie_id + 1  # Ensure that the item IDs fit within the embedding layer

# Split data into train and test sets
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# Neural Collaborative Filtering model
def NCF_model(n_users, n_items, embedding_size=50):
    user_input = Input(shape=(1,))
    item_input = Input(shape=(1,))
    
    user_embedding = Embedding(n_users, embedding_size, input_length=1)(user_input)
    item_embedding = Embedding(n_items, embedding_size, input_length=1)(item_input)
    
    user_vec = Flatten()(user_embedding)
    item_vec = Flatten()(item_embedding)
    
    concat = Concatenate()([user_vec, item_vec])
    
    dense = Dense(128, activation='relu')(concat)
    output = Dense(1)(dense)
    
    model = models.Model([user_input, item_input], output)
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    
    return model

model = NCF_model(n_users, n_items)
model.summary()

# Train the model
history = model.fit([train_data['userId'], train_data['movieId']], train_data['rating'], 
                    epochs=5, batch_size=64, 
                    validation_data=([test_data['userId'], test_data['movieId']], test_data['rating']))

# Evaluate the model
predictions = model.predict([test_data['userId'], test_data['movieId']])
rmse = np.sqrt(mean_squared_error(test_data['rating'], predictions))
print(f'RMSE for NCF: {rmse}')
