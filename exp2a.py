import argparse

import numpy as np
import lmdb
import os
import pickle
import h5py
import math

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
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical

BATCH_SIZE = 20
DEFAULT_MODEL_PATH = 'exp2a_model.h5'
DEFAULT_NUM_EPOCH = 40
NUM_INGRE = 1048 # TODO: know what number this is

def new_model():
  # See https://keras.io/applications/#fine-tune-inceptionv3-on-a-new-set-of-classes
  resnet = ResNet50(input_shape=(3,256,256), pooling='avg', include_top=False, weights='imagenet')
  # Freeze resnet layers
  for layer in resnet.layers:
    layer.trainable = False
  x = resnet.output
  x = Dense(NUM_INGRE, activation='linear')(x)
  x = Dropout(0.2)(x)
  ingre_category = Dense(NUM_INGRE, activation='softmax', name='ingre_category')(x)

  y = keras.layers.concatenate([resnet.output, ingre_category])
  y = Dense(1048, activation='linear')(y)
  y = Dropout(0.2)(y)
  food_category = Dense(1048, activation='softmax', name='food_category')(y)

  return Model(inputs=resnet.input, outputs=[ingre_category, food_category])

def open_model(filename):
  return load_model(filename)

def train(model_path=None, num_epoch=DEFAULT_NUM_EPOCH):
  # Set directory for model load and save:
  if model_path is None:
    model_path = DEFAULT_MODEL_PATH
  if os.path.exists(model_path):
    model = open_model(model_path)
  else:
    model = new_model()

  # Set default directories
  data_path = 'data/'
  keys_path = 'data/test_keys.pkl'
  images_path = 'data/images/test'
  classes_path = 'data/classes1M.pkl'
  db_path = 'data/data.h5'

  # classes_file = open(classes_path, 'rb')
  # id2class = pickle.load(classes_file)
  # ind2class = pickle.load(classes_file)

  db = h5py.File(db_path, 'r')

  model.compile(loss={'ingre_category': 'categorical_crossentropy',
                      'food_category': 'categorical_crossentropy'},
                optimizer='SGD', metrics=['accuracy'])
  model.fit_generator(
    batch_generator(db, batch_size=BATCH_SIZE,partition='train'),
    validation_data=batch_generator(db, batch_size=BATCH_SIZE, partition='val'),
    steps_per_epoch=math.floor(238459 / BATCH_SIZE),
    validation_steps=math.floor(51129 / BATCH_SIZE),
    epochs=num_epoch,
    verbose=1
  )

  model.save(model_path)

def batch_generator(db, batch_size=100, partition='train'):
  # TODO: add ingredient result in output, format: outputs=[ingre_category, food_category]
  ids = db['ids_{}'.format(partition)]
  classes = db['classes_{}'.format(partition)]
  impos = db['impos_{}'.format(partition)]
  ims = db['ims_{}'.format(partition)]
  numims = db['numims_{}'.format(partition)]
  partition_size = len(ids)
  while(True):
    images = []
    categories = []
    for i in np.random.choice(partition_size, size=batch_size): 
      id = ids[i]
      # Since category=0 is the background class, we can ignore that
      # Experiment with ignoring category 1 (peanut butter, comprising half of
      # all)
      category = classes[i]
      if category == 0 or category == 1:
        continue
      category = category - 1
      for j in range(numims[i]):
        index = impos[i][j] - 1
        image = ims[index]
        images.append(image)
        categories.append(category)

    batch_x = np.array(images)
    batch_y = to_categorical(categories, 1048)
    # TODO: return proper ingre_category
    yield (batch_x, {'ingre_category': batch_y, 'food_category': batch_y})


def test():
  model = load_model(DEFAULT_MODEL_PATH)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Baseline NN')
  parser.add_argument('-p', '--model_path', dest='model_path')
  parser.add_argument('-e', '--num_epoch', dest='num_epoch', default=DEFAULT_NUM_EPOCH, type=int)
  args = parser.parse_args()
  train(**vars(args))