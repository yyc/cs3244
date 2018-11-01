"""
Read output of model and perfomr similarity calculation
"""

import argparse
import ijson as ijs
import json
from keras.layers import Input, Embedding, Dot, Reshape, Dense
from keras.models import Model, load_model
import numpy as np
import pandas as pd

FOOD_EMBEDDING_LAYER_NAME = "food_embedding"

def extract_food_weights_from_model(food_ingredient_model):
    food_layer = food_ingredient_model.get_layer(FOOD_EMBEDDING_LAYER_NAME)
    food_weights = food_layer.get_weights()[0]

    weights = food_weights / np.linalg.norm(food_weights, axis = 1).reshape((-1, 1))

    return weights

#with open(JSON_DUMP_PATH, 'w') as outfile:
#    json.dump(pd.DataFrame(food_weights).to_dict(), outfile)

#INDEX_TO_FIND = 0
#dists = np.dot(food_weights, food_weights[INDEX_TO_FIND])
#sorted_dists = np.argsort(-dists)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Processor')
    parser.add_argument('-m', '--model_path', dest='model_path')
    args = parser.parse_args()

    food_ingredient_model = load_model(args["model_path"])
    food_ingredient_model.summary()

    food_weights = extract_food_weights_from_model(food_ingredient_model)

    # Given 2 arrays of ids to compare, return their similarity index

    list1 =
    list2 =

    z
