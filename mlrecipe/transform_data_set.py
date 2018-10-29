import ijson as ijs
import json
import numpy
import pandas as pd
import sys

#####################################
# Config                            #
#####################################
DATA_FILENAME = "det_ingrs.json"

FOOD_INGREDIENTS_KEY = 'ingredients'
FOOD_ID_KEY = 'id'
INGREDIENT_TEXT_KEY = 'text'

###############

TRAINING_PACKET_NEGATIVE_RATIO = 1 # If 2, means Positive:Negative = 1/2

#####################################
# Part 1: Parsing json file         #
#####################################
count = 1

pd_data_values  = []
pd_data_columns = ['food', 'ingredient','train']

#TRAINING_PACKET_FILENAME = "full_training_packet_part2.json"
#RANGE_LB = 100000
#RANGE_UB = 199999
TRAINING_PACKET_FILENAME = sys.argv[1]
RANGE_LB = int(sys.argv[2])
RANGE_UB = int(sys.argv[3])



with open(DATA_FILENAME, 'r') as f:
    food = ijs.items(f, 'item')

    for x in food:

        if count > RANGE_UB:
            break

        if count >= RANGE_LB:
            row = json.loads(json.dumps(x))
            food_id = row[FOOD_ID_KEY]
            ingredients_list = [[food_id, i[INGREDIENT_TEXT_KEY], 1] for i in row[FOOD_INGREDIENTS_KEY]]

            print count

            pd_data_values.extend(ingredients_list)

        count = count + 1

food_compositions = pd.DataFrame(pd_data_values, columns=pd_data_columns)

###########################################
# Part 2: Splitting into training packets #
###########################################

num_positives = (count * 2)/(1+TRAINING_PACKET_NEGATIVE_RATIO)
num_negatives = (count * 2) - num_positives
#import pdb; pdb.set_trace()
training_packet = []

food_positives = food_compositions.head(num_positives).copy(True)

###### HACK SPEED UP
current_food_fid = None
current_foods_positive = []
#####################

# Generate negatives
constructed_negatives = []

for i, r in food_positives.iterrows():
    fid = r['food']

    if current_food_fid != fid:
        # Fetch list of correct food and ingredient
        current_foods_positive = food_compositions[(food_compositions['food'] == fid)]
        current_food_id = fid
    while True:
        not_ingredient = food_compositions.sample()['ingredient'].iloc[0]

        if current_foods_positive[(current_foods_positive['food'] == fid) &
                                  (current_foods_positive['ingredient'] == not_ingredient)].empty:
            break

    neg = [fid, not_ingredient, 0]
    print "#{}. [{}, {}, 0]".format(i, fid, not_ingredient)
    constructed_negatives.append(neg)

constructed_negatives = pd.DataFrame(constructed_negatives, columns=pd_data_columns)
training_packet = food_positives.copy(True)
training_packet = training_packet.append(constructed_negatives)
training_packet = training_packet.sample(frac=1).reset_index(drop=True)

print("Time to write")

with open(TRAINING_PACKET_FILENAME, 'w') as outfile:
    json.dump(training_packet.values.tolist(), outfile)
