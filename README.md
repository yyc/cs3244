# Creating embeddings for similarity lookup between ingredient lists

See [repo](https://github.com/yyc/cs3244) first.

*Objective*: To create a NN/lookup that can provide a similarity score for food
based on their ingredients. This is part of our project to extend this
[project](http://pic2recipe.csail.mit.edu/).

## Derek's Plan

1. [X] Understand embeddings
2. [X] Understand Keras
3. [X] Tidy Data
4. [ ] Construct NN
5. [ ] Run NN
5. [ ] Sanity check on results
6. [ ] Repeat above with more data
7. [ ] Integrate with NNs done by the rest

## 0. Finding data

We will be using the dataset from the **Recipe1M** project, with over 1m cooking
recipes and 800k food images. More specifically, we will be using
`det_ingrs.json` which contains information about the ingredients that appear in
each prepared food.

The format for `det_ingrs.json` is as follows:

```javascript
[  
   {  
      "valid":[  ],
      "id":"000018c8a5",
      "ingredients":[  
         {  
            "text":"penne"
         },
         {  
            "text":"cheese sauce"
         },
         {  
            "text":"cheddar cheese"
         },
         ...]
   }, ...
]
```

## 1. Transforming `det_ingrs.json` into training data

Since the size of raw data is 361085654 bytes, I will be splitting json file into smaller chunks to process them.

Using a bash script (see `split_det_ingrs.sh`), split `det_ingrs.json` into
smaller files. Note that json structures will be broken for the smaller files
and they will be manually edited.

There might be an easier way to achieve this, but this dirty method works for memory poor me.

## 2. For each ingredient object, split them into our training data format.

Training data format will be:
`(name of food, ingredient in food, whether ingredient is in food)`

We want both positive and negative training examples, and thus we label the
combinations with `1` or `0`.

~~Plan A: I decided to use a tiny database [TinyDB](https://pypi.org/project/tinydb/) to store the training data created. I don't want to overheat my MBP, which is possible if I were to load the entire file into memory.~~

Plan B: Use `ijson` to stream json data, use `pandas` to visualise the data for sanity check, create a function that can return a random set of data for training.

### Sanity check

Using `print food_compositions.groupby('ingredient')[u'food'].nunique()` on a small segment of data:

```
non - fat vanilla yogurt                                          1
olive oil                                                         1
penne                                                             1
pimentos                                                          1
pineapple chunks                                                  1
raw zucchini                                                      1
red bell pepper                                                   1
red onion                                                         1
salt                                                              2
salt and black pepper                                             1
seedless watermelon                                               1
shredded coconut                                                  1
soy sauce                                                         1
strawberries                                                      1
strawberry Jell - O gelatin dessert                               1
sugar                                                             1
tea                                                               1
tomatoes                                                          1
```

and `food_compositions.groupby('ingredient')['food'].nunique().sort_values(ascending=False).reset_index(name='count')`

```
0 salt 3294
1 butter 2116
2 sugar 1850
3 olive oil 1487
4 water 1457
5 eggs 1437
6 garlic cloves 1241
7 onion 973
8 milk 969
9 flour 912
10 all - purpose flour 908
11 onions 855
12 egg 810
13 baking powder 691
14 pepper 686
15 brown sugar 652
16 unsalted butter 646
17 vegetable oil 630
18 vanilla extract 594
19 baking soda 580
20 salt and pepper 557
21 lemon juice 513
22 parmesan cheese 500
23 garlic clove 497
24 black pepper 480
25 tomatoes 477
26 vanilla 473
27 garlic 456
28 honey 446
29 carrots 444
```
**Observation:** Data is not entirely clean, there are ingredients that need to be removed/edited.

Also, since reading `1000000` food rows takes about `486` seconds on my computer (Intel(R) Core(TM) i5-4460S CPU @ 2.90GHz), we have decided to split training data into packets of `1000000` positives and negatives.

### Preparing Training Packets

#### Mixing round 1

Each training packet will contain `50%` of data from json, and `50%` from negative training inputs.

See `prepare_training_packet.py` for implementation.

#### Mixing round 2

**Future**

We will intersperse training data around the different training packets to see how training performance varies.

## 3. Train a classifier

** TODO, NOT DONE **

Sketch:

Input will be food and ingredient.

Every epoch, load a set of training examples and run them

There should be a general decreasing loss result.

Train NN classifier until low loss is seen.

Save NN, extract embeddings

## 4. Test results


## References
* https://keras.io/#you-have-just-found-keras
* https://blog.keras.io/category/tutorials.html
* https://medium.com/google-cloud/keras-inception-v3-on-google-compute-engine-a54918b0058
* https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
* https://github.com/WillKoehrsen/wikipedia-data-science/blob/master/notebooks/Book%20Recommendation%20System.ipynb
* https://towardsdatascience.com/deep-learning-4-embedding-layers-f9a02d55ac12
* https://stats.stackexchange.com/questions/270546/how-does-keras-embedding-layer-work
* https://medium.com/@satnalikamayank12/on-learning-embeddings-for-categorical-data-using-keras-165ff2773fc9
* https://www.dataquest.io/blog/python-json-tutorial/
