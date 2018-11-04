"""
Query model for foodsimilarity
"""
import argparse
import ijson as ijs
import json
from keras.layers import Input, Embedding, Dot, Reshape, Dense
from keras.models import Model, load_model
import numpy as np
import pandas as pd
import pickle

class FoodSimilarityQuery:
    def __init__(self, model_path, food_id_to_int_path):
        food_ingredient_model = load_model(model_path)
        food_ingredient_model.summary()

        self.food_embedding_layer_name = "food_embedding"

        self.__extract_food_weights_from_model(food_ingredient_model)
        self.__load_food_id_to_int(food_id_to_int_path)

    def __extract_food_weights_from_model(self, food_ingredient_model):

        food_layer = food_ingredient_model.get_layer(self.food_embedding_layer_name)
        food_weights = food_layer.get_weights()[0]

        self.weights = food_weights / np.linalg.norm(food_weights, axis = 1).reshape((-1, 1))

    def __load_food_id_to_int(self, path):
        print("Loading {}".format(path))
        self.food_id_to_int = pickle.load(open(path, "rb"))
        print("Loading done!")

    def check(self, id1, id2):
        try:
            index1 = self.food_id_to_int[id1]
            index2 = self.food_id_to_int[id2]

            return np.dot(self.weights[index1], self.weights[index2])

        except KeyError as e:
            print("Illegal index! Heckin bork! {}".format(e.message))

    def get_embedding(self, fid):
        try:
            ind = self.food_id_to_int[fid]

            return self.weights[ind]

        except KeyError as e:
            print("Illegal index! Heckin bork! {}".format(e.message))
        #INDEX_TO_FIND = 0
	    #dists = np.dot(food_weights, food_weights[INDEX_TO_FIND])
	    #sorted_dists = np.argsort(-dists)

if __name__ == '__main__':
    x = FoodSimilarityQuery("models/embedding-0.99.h5", "food_id_to_int.p")

    import pdb; pdb.set_trace()
