import argparse

import numpy as np
import os
import pickle
import h5py
import math

from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras_preprocessing.image import ImageDataGenerator
from keras import backend as K

from mlrecipe.food_similarity_query import FoodSimilarityQuery

# Use GPU 1 since other people are using 0
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

np.random.seed(123)

# limit memory usage to play nice with others
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

import keras
keras.backend.set_image_data_format("channels_first")

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Flatten, Dropout
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

BATCH_SIZE = 20
DEFAULT_MODEL_PATH = 'exp2a_ingre_model.h5'
DEFAULT_NUM_EPOCH = 40
NUM_INGRE = 50 # Embedding of size 50

# Helper for cosine similarity
def cos_distance(y_true, y_pred):
  y_true = K.l2_normalize(y_true, axis=-1)
  y_pred = K.l2_normalize(y_pred, axis=-1)
  return K.mean(1 - K.sum((y_true * y_pred), axis=-1))


def new_model():
  # See https://keras.io/applications/#fine-tune-inceptionv3-on-a-new-set-of-classes
  resnet = ResNet50(input_shape=(3,256,256), pooling='avg', include_top=False, weights='imagenet')
  # Freeze resnet layers
  for layer in resnet.layers:
    layer.trainable = False
  x = resnet.output
  x = Dense(200, activation='relu')(x)
  x = Dropout(0.2)(x)
  x = Dense(100, activation='relu')(x)
  x = Dropout(0.2)(x)
  x = Dense(NUM_INGRE, activation='relu')(x)
  x = Dropout(0.2)(x)
  ingre_category = Dense(NUM_INGRE, activation='softmax', name='ingre_category')(x)

  return Model(inputs=resnet.input, outputs=ingre_category)

def open_model(filename):
  return load_model(filename)

def train(model_path=None, num_epoch=DEFAULT_NUM_EPOCH, checkpoint_path=None, learning_rate = 0.01, momentum=0.9, augment=False, weights_path=None):
  # Set directory for model load and save:
  if model_path is None:
    model_path = DEFAULT_MODEL_PATH

  if weights_path and os.path.exists(weights_path):
    model = new_model()
    model.load_weights(weights_path)
    print("Initialized new model and loaded weights from {}".format(weights_path))
  elif os.path.exists(model_path):
    model = open_model(model_path)
    print("Opened existing model {}".format(model_path))
  else:
    model = new_model()
    print("initialized new model")

  # Set default directories
  data_path = 'data/'
  keys_path = 'data/test_keys.pkl'
  images_path = 'data/images/test'
  classes_path = 'data/classes1M.pkl'
  db_path = 'data/subset_train_val.h5'
  # db_path = '/home/thamsy/Downloads/subset_train_val.h5'

  # classes_file = open(classes_path, 'rb')
  # id2class = pickle.load(classes_file)
  # ind2class = pickle.load(classes_file)

  db = h5py.File(db_path, 'r')
  fsq = FoodSimilarityQuery('models/embedding-1.00.h5', 'mlrecipe/food_id_to_int.p')

  callbacks = []
  if checkpoint_path is not None:
    checkpointer = ModelCheckpoint(checkpoint_path, monitor="val_food_category_acc", mode="max", save_best_only=True, save_weights_only=True)
    callbacks.append(checkpointer)

  optimizer = SGD(lr=learning_rate, momentum=momentum)

  model.compile(loss=cos_distance,
                # loss_weights={'ingre_category': 1,
                #       'food_category': 9},
                optimizer=optimizer, metrics=['accuracy'])
  model.fit_generator(
    batch_generator(db, fsq, batch_size=BATCH_SIZE,partition='train', augment=augment),
    validation_data=batch_generator(db, fsq, batch_size=BATCH_SIZE, partition='val'),
    steps_per_epoch=math.floor(238459 / BATCH_SIZE),
    validation_steps=math.floor(51129 / BATCH_SIZE),
    epochs=num_epoch,
    verbose=1,
    callbacks = callbacks
  )

  model.save(model_path)

def batch_generator(db, fsq, batch_size=100, partition='train', augment=False):
  # TODO: add ingredient result in output, format: outputs=[ingre_category, food_category]
  ids = db['ids_{}'.format(partition)]
  classes = db['classes_{}'.format(partition)]
  impos = db['impos_{}'.format(partition)]
  ims = db['ims_{}'.format(partition)]
  numims = db['numims_{}'.format(partition)]
  # ingrs = db['ingrs_{}'.format(partition)]
  partition_size = len(ids)
  if augment:
      augmenter = ImageDataGenerator(
          data_format="channels_first",
          horizontal_flip=True,
          zoom_range=0.1,
          width_shift_range=0.2,
          height_shift_range=0.2,
          fill_mode="constant"
      )
      im_shape = ims[0].shape
      random_augmenters = [augmenter.get_random_transform(im_shape) for i in range(1000)]
      print("Created augmenter")
  while(True):
    images = []
    ingredients = []
    while(len(images) < batch_size):
      # i in np.random.choice(partition_size, size=batch_size):
      i = np.random.random_integers(0, partition_size - 1)
      id = ids[i].decode("utf-8")
      # Since category=0 is the background class, we can ignore that
      # Experiment with ignoring category 1 (peanut butter, comprising half of
      # all)
      category = classes[i]
      ingr = fsq.get_vector_for_food(id)
      if len(ingr) == 0:
        continue
      for j in range(numims[i]):
        index = impos[i][j] - 1
        # Only augment 1/5 of the time
        if augment and np.random.randint(5) >= 4:
          opts = np.random.choice(random_augmenters, 1)[0]
          image = augmenter.apply_transform(ims[index], opts)
        else:
          image = ims[index]
        images.append(image)
        ingredients.append(ingr)
    batch_x = np.array(images)

    yield (batch_x, np.array(ingredients))


def test():
  model = load_model(DEFAULT_MODEL_PATH)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Baseline NN')
  parser.add_argument('-p', '--model_path', dest='model_path')
  parser.add_argument('-e', '--num_epoch', dest='num_epoch', default=DEFAULT_NUM_EPOCH, type=int)
  parser.add_argument('-c', '--checkpoint', dest='checkpoint_path')
  parser.add_argument('-w', '--weights', dest="weights_path")
  parser.add_argument('-a', '--augment', dest='augment', default=False, type=bool)
  parser.add_argument('-lr', dest='learning_rate', type=float, default=0.02)
  parser.add_argument('-m', dest='momentum', type=float, default=0.9)
  args = parser.parse_args()
  train(**vars(args))
