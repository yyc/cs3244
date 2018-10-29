"""
Read embeddings json and calculate cosine difference
"""

import json
import numpy as np
import pandas as pd

JSON_DUMP_PATH = "json/embeddings.json"

food_weights = {}

with open(JSON_DUMP_PATH) as json_data:
    d = json.load(json_data)
    food_weights = pd.DataFrame(d)

INDEX_TO_FIND = 0

dists = np.dot(food_weights,food_weights.iloc[INDEX_TO_FIND])
sorted_dists = np.argsort(-dists)

import pdb; pdb.set_trace()
