import ijson as ijs
import json
import pandas as pd

datafilename = "det_ingrs.json"

FOOD_INGREDIENTS_KEY = 'ingredients'
FOOD_ID_KEY = 'id'
INGREDIENT_TEXT_KEY = 'text'

FOOD_PARSING_LIMIT = 1000000

count = 1

pd_data_values  = []
pd_data_columns = ['food', 'ingredient']

with open(datafilename, 'r') as f:
    food = ijs.items(f, 'item')

    for x in food:
        row = json.loads(json.dumps(x))
        food_id = row[FOOD_ID_KEY]
        ingredients_list = [[food_id, i[INGREDIENT_TEXT_KEY]] for i in row[FOOD_INGREDIENTS_KEY]]

        print count

        pd_data_values.extend(ingredients_list)

        count = count + 1

        if count > FOOD_PARSING_LIMIT:
            break

food_compositions = pd.DataFrame(pd_data_values, columns=pd_data_columns)
print food_compositions.groupby('food')['ingredient'].nunique()
#print food_compositions.groupby('ingredient')['food'].nunique().sort_values(ascending=False).reset_index(name='count')

#print food_compositions.groupby('food')['ingredient'].nunique()
