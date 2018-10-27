"""
Read output of model and output to a json its weights
"""

import ijson as ijs
import json
from keras.layers import Input, Embedding, Dot, Reshape, Dense
from keras.models import Model, load_model
import numpy as np
import pandas as pd

JSON_DUMP_PATH = "json/embeddings.json"

MODEL_PATH = "models/model.h5"
FOOD_EMBEDDING_LAYER_NAME = "food_embedding"

food_ingredient_model = load_model(MODEL_PATH)
food_ingredient_model.summary()

food_layer = food_ingredient_model.get_layer(FOOD_EMBEDDING_LAYER_NAME)
food_weights = food_layer.get_weights()[0]
food_weights.shape

food_weights = food_weights / np.linalg.norm(food_weights, axis = 1).reshape((-1, 1))

with open(JSON_DUMP_PATH, 'w') as outfile:
    json.dump(pd.DataFrame(food_weights).to_dict(), outfile)

#INDEX_TO_FIND = 0
#dists = np.dot(food_weights, food_weights[INDEX_TO_FIND])
#sorted_dists = np.argsort(-dists)

import pdb; pdb.set_trace()
