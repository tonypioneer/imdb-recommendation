import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, Flatten, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load data
ratings = pd.read_csv('../data/input/ratings.csv')
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
ratings['hour'] = ratings['timestamp'].dt.hour
ratings['day_of_week'] = ratings['timestamp'].dt.dayofweek

# Check the maximum userId and movieId in the dataset
max_user_id = ratings['userId'].max()
max_movie_id = ratings['movieId'].max()

n_users = max_user_id + 1  # Ensure that the user IDs fit within the embedding layer
n_items = max_movie_id + 1  # Ensure that the item IDs fit within the embedding layer

# Split data into train and test sets
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# Define the context-aware NCF model
def context_aware_NCF_model(n_users, n_items, embedding_size=50):
    user_input = Input(shape=(1,))
    item_input = Input(shape=(1,))
    hour_input = Input(shape=(1,))
    day_of_week_input = Input(shape=(1,))
    
    user_embedding = Embedding(n_users, embedding_size, input_length=1)(user_input)
    item_embedding = Embedding(n_items, embedding_size, input_length=1)(item_input)
    
    user_vec = Flatten()(user_embedding)
    item_vec = Flatten()(item_embedding)
    
    hour_vec = Embedding(24, embedding_size, input_length=1)(hour_input)
    day_of_week_vec = Embedding(7, embedding_size, input_length=1)(day_of_week_input)
    
    hour_vec = Flatten()(hour_vec)
    day_of_week_vec = Flatten()(day_of_week_vec)
    
    concat = Concatenate()([user_vec, item_vec, hour_vec, day_of_week_vec])
    
    dense = Dense(128, activation='relu')(concat)
    output = Dense(1)(dense)
    
    model = Model([user_input, item_input, hour_input, day_of_week_input], output)
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    
    return model

model = context_aware_NCF_model(n_users, n_items)
model.summary()

# Train the context-aware model
history = model.fit([train_data['userId'], train_data['movieId'], train_data['hour'], train_data['day_of_week']], 
                    train_data['rating'], 
                    epochs=5, batch_size=64, 
                    validation_data=([test_data['userId'], test_data['movieId'], test_data['hour'], test_data['day_of_week']], 
                                     test_data['rating']))

# Evaluate the context-aware model
predictions = model.predict([test_data['userId'], test_data['movieId'], test_data['hour'], test_data['day_of_week']])
rmse = np.sqrt(mean_squared_error(test_data['rating'], predictions))
mae = mean_absolute_error(test_data['rating'], predictions)
print(f'RMSE for context-aware NCF: {rmse}')
print(f'MAE for context-aware NCF: {mae}')
