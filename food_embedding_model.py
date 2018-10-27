import ijson as ijs
import json
from keras.layers import Input, Embedding, Dot, Reshape, Dense
from keras.models import Model
import pandas as pd

######################################
import FoodIngredientRecorder as FoodIngredientRecorder
recorder = FoodIngredientRecorder.FoodIngredientRecorder()
#####################################################

pd_data_values  = []
pd_data_columns = ['food_id', 'ingredient_id','train']

TRAINING_PACKET_FILENAME = "full_training_packet_part2.json"

with open(TRAINING_PACKET_FILENAME, 'r') as f:
    food = ijs.items(f, 'item')
    count = 0
    for x in food:
        try:
            row = json.loads(json.dumps(x))
            print row
            transformed_row = recorder.process_and_record(row[0], row[1])
            transformed_row.append(row[2])
            print transformed_row
            pd_data_values.extend([transformed_row])
        except UnicodeEncodeError:
            None
        count = count + 1

food_compositions = pd.DataFrame(pd_data_values, columns=pd_data_columns)

################################

NUM_EMBEDDINGS = 50

food = Input(name = 'food', shape = [1])
ingredient = Input(name = 'ingredient', shape = [1])

num_inputs = count #

food_embedding = Embedding(name='food_embedding',
                           input_dim=len(recorder.get_food_int_to_id()),
                           output_dim=NUM_EMBEDDINGS)(food)

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

print model.summary()

#food_id = food_compositions['food'].apply(lambda x: 1)
food_col = food_compositions[[pd_data_columns[0]]]
ingredient_col = food_compositions[[pd_data_columns[1]]]
correct_output_col = food_compositions.iloc[:,-1]
model.fit(x=[food_col, ingredient_col], y=correct_output_col, verbose=2, epochs=15, steps_per_epoch=1000)

model.save('./models/model.h5')
