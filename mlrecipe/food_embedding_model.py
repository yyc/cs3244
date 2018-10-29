import json
import pickle
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Embedding, Dot, Reshape, Dense
from keras.models import Sequential, Model, load_model
import pandas as pd
import argparse
import os
import math
import numpy as np

import FoodIngredientRecorder as FoodIngredientRecorder

NUM_EMBEDDINGS = 50
TRAINING_PACKET_FILENAME = "full_training.json"
BATCH_SIZE=1000

np.random.seed(123)


# Use GPU 1 since other people are using 0
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# limit memory usage to play nice with others
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

def fetch_recorder(filename):

    recorder = FoodIngredientRecorder.FoodIngredientRecorder()

    pd_data_columns = ['food_id', 'ingredient_id','train']

    with open(filename, 'rb') as f:
        food = pickle.load(f)
        pd_data_values = [
            recorder.process_and_record(row[0], row[1]) + [row[2]]
            for row in food
        ]

    compositions = pd.DataFrame(pd_data_values, columns=pd_data_columns)
    return recorder, compositions

################################

def get_model(recorder):
    food = Input(name = 'food', shape = [1])
    ingredient = Input(name = 'ingredient', shape = [1])

    food_embedding = Embedding(
                        name='food_embedding',
                        input_dim=len(recorder.get_food_int_to_id()),
                        output_dim=NUM_EMBEDDINGS
                        )(food)

    ingredient_embedding = Embedding(name='ingredient_embedding',
                                     input_dim=len(recorder.get_ingredient_int_to_name()),
                                     output_dim=NUM_EMBEDDINGS)(ingredient)

    food_ingredient = Dot(name="food_ingredient_dot",
                          normalize=True,
                          axes=2)([food_embedding, ingredient_embedding])

    combination = Reshape(target_shape = [1])(food_ingredient)

    merged = Dense(1, activation = 'sigmoid')(combination)

    model = Model(inputs = [food, ingredient], outputs = merged)
    model.compile(optimizer='SGD', loss = 'binary_crossentropy', metrics = ['accuracy'])

    print(model.summary())

    return model


def train(filename=TRAINING_PACKET_FILENAME, model_path='./models/model.h5', num_epoch=10, batch_size=BATCH_SIZE):
    recorder, food_compositions = fetch_recorder(filename)
    cp_callback = ModelCheckpoint("models/embedding{epoch:02d}-{train_loss:.2f}.h5",
                                  save_weights_only=False,
                                  verbose=1)
    # if os.path.exists(model_path):
    #     model = load_model(model_path)
    # else:
    model = get_model(recorder)

    columns = food_compositions.columns

    # food_id = food_compositions['food'].apply(lambda x: 1)
    food_col = food_compositions[[columns[0]]]
    ingredient_col = food_compositions[[columns[1]]]
    correct_output_col = food_compositions.iloc[:, -1]

    count = len(food_col)

    model.fit_generator(
        training_generator(batch_size, food_col, ingredient_col, correct_output_col),
        verbose=1,
        epochs=num_epoch,
        steps_per_epoch=math.ceil(count / batch_size),
        callbacks=[cp_callback]
    )

    model.save(model_path)


def training_generator(batch_size, food, ingredients, correct):
    count = len(food)
    arr = list(range(count))
    while(True):
        np.random.shuffle(arr)
        for start in range(0, count, batch_size):
            sample = arr[start:start+batch_size]
            yield(
                [np.take(food, sample, axis=0), np.take(ingredients, sample, axis=0)],
                np.take(correct,sample, axis=0)
            )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline NN')
    parser.add_argument('-p', '--model_path', dest='model_path')
    parser.add_argument('-e', '--num_epoch', dest='num_epoch', type=int)
    parser.add_argument('-f', '--data_file', dest='filename')
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int)
    args = parser.parse_args()
    train(**vars(args))
