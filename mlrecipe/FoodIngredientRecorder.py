"""
Stores food and ingredient for easy lookup
"""
import pickle

class FoodIngredientRecorder:
    def __init__(self):
        # food comes originally with an id
        # our internal id is represented by int
        self.food_int_to_id = {}
        self.food_id_to_int = {}

        self.fcount = 0
        self.icount = 0

        self.ingredient_int_to_name = {}
        self.ingredient_name_to_int = {}

    def save(self):
        food_int_to_id_path = "food_int_to_id.p"
        food_id_to_int_path = "food_id_to_int.p"
        ingredient_int_to_name_path = "ingredient_int_to_name.p"
        ingredient_name_to_int_path = "ingredient_name_to_int.p"

        pickle.dump(self.food_int_to_id, open(food_int_to_id_path, "wb"))
        pickle.dump(self.food_id_to_int, open(food_id_to_int_path, "wb"))
        pickle.dump(self.ingredient_int_to_name, open(ingredient_int_to_name_path, "wb"))
        pickle.dump(self.ingredient_name_to_int, open(ingredient_name_to_int_path, "wb"))

        print("saved")

    def process_and_record(self, fid, iname):
        # print("fid:{} iname: {}".format(fid, iname))
        if fid not in self.food_id_to_int:

            # Add new mapping from food_id to int
            self.food_id_to_int[fid] = self.fcount

            # Add new mapping from int to food_id
            self.food_int_to_id[self.fcount] = fid

            self.fcount = self.fcount + 1

        if iname not in self.ingredient_name_to_int:
            # Add new mapping from ingredient_name to int
            self.ingredient_name_to_int[iname] = self.icount

            # Add new mapping from int to ingredient name
            self.ingredient_int_to_name[self.icount] = iname

            self.icount = self.icount + 1
        return [self.food_id_to_int[fid], self.ingredient_name_to_int[iname]]
    def get_food_id_to_int(self):
        return self.food_id_to_int

    def get_food_int_to_id(self):
        return self.food_int_to_id

    def get_ingredient_int_to_name(self):
        return self.ingredient_int_to_name

    def get_ingredient_name_to_int(self):
        return self.ingredient_name_to_int
