import ijson as ijs
import json
import numpy
import pandas as pd

#####################################
# Config                            #
#####################################
DATA_FILENAME = "det_ingrs.json"

FOOD_INGREDIENTS_KEY = 'ingredients'
FOOD_ID_KEY = 'id'
INGREDIENT_TEXT_KEY = 'text'

FOOD_PARSING_LIMIT = 1000

###############

TRAINING_PACKET_FILENAME = "training_packet_1.json" # TODO change to string template
TRAINING_PACKET_SIZE = 2000
TRAINING_PACKET_NEGATIVE_RATIO = 1 # If 2, means Positive:Negative = 1/2

#####################################
# Part 1: Parsing json file         #
#####################################
count = 1

pd_data_values  = []
pd_data_columns = ['food', 'ingredient','train']

with open(DATA_FILENAME, 'r') as f:
    food = ijs.items(f, 'item')

    for x in food:
        row = json.loads(json.dumps(x))
        food_id = row[FOOD_ID_KEY]
        ingredients_list = [[food_id, i[INGREDIENT_TEXT_KEY], 1] for i in row[FOOD_INGREDIENTS_KEY]]

        print count

        pd_data_values.extend(ingredients_list)

        count = count + 1

        if count > FOOD_PARSING_LIMIT:
            break

food_compositions = pd.DataFrame(pd_data_values, columns=pd_data_columns)

###########################################
# Part 2: Splitting into training packets #
###########################################

num_positives = TRAINING_PACKET_SIZE/(1+TRAINING_PACKET_NEGATIVE_RATIO)
num_negatives = TRAINING_PACKET_SIZE - num_positives

training_packet = []

food_positives = food_compositions.head(num_positives).copy(True)

# Generate negatives
constructed_negatives = []

for i, r in food_positives.iterrows():
    fid = r['food']
    while True:
        not_ingredient = food_compositions.sample()['ingredient'].iloc[0]

        if food_compositions[(food_compositions['food'] == fid) &
                             (food_compositions['ingredient'] == not_ingredient)].empty:
            break

    neg = [fid, not_ingredient, 0]
    constructed_negatives.append(neg)

constructed_negatives = pd.DataFrame(constructed_negatives, columns=pd_data_columns)
training_packet = food_positives.copy(True)
training_packet = training_packet.append(constructed_negatives)
training_packet = training_packet.sample(frac=1).reset_index(drop=True)

with open(TRAINING_PACKET_FILENAME, 'w') as outfile:
    json.dump(training_packet.values.tolist(), outfile)
