import ijson as ijs
import json
import pandas as pd

datafilename = "det_ingrs.json"

FOOD_INGREDIENTS_KEY = 'ingredients'
FOOD_ID_KEY = 'id'
INGREDIENT_TEXT_KEY = 'text'

count = 1

food_composition = pd.DataFrame

pd_data_values  = []
pd_data_columns = ['food', 'ingredient']

with open(datafilename, 'r') as f:
    food = ijs.items(f,'item')

    for x in food:
        row = json.loads(json.dumps(x))
        food_id = row[FOOD_ID_KEY]
        ingredients_list = [[food_id,i[INGREDIENT_TEXT_KEY]] for i in row[FOOD_INGREDIENTS_KEY]]
        
        print count
        print food_id
        print ingredients_list

        pd_data_values.extend(ingredients_list)

        count = count + 1

        if count > 10:
            break

