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
from keras.callbacks import ModelCheckpoint

BATCH_SIZE = 20
DEFAULT_MODEL_PATH = 'baseline_model.h5'
DEFAULT_NUM_EPOCH = 40

def new_model():
  resnet = ResNet50(input_shape=(3,256,256), pooling='avg', include_top=False, weights='imagenet')
  # Freeze resnet layers
  for layer in resnet.layers:
    layer.trainable = False
  top_model = Sequential()
  top_model.add(Dense(1048, activation='linear'))
  top_model.add(Dropout(0.2))
  top_model.add(Dense(1048, activation='softmax'))
  model = Model(inputs=resnet.input, outputs=top_model(resnet.output))
  return model

def open_model(filename):
  return load_model(filename)

def train(model_path=None, num_epoch=DEFAULT_NUM_EPOCH, checkpoint_path=None):
  # Set directory for model load and save:
  if model_path is None:
    model_path = DEFAULT_MODEL_PATH
  if os.path.exists(model_path):
    model = open_model(model_path)
    print("loaded model {}".format(model_path))
  else:
    model = new_model()
    print("initialized new model")

  if num_epoch is None:
    num_epoch = DEFAULT_NUM_EPOCH

  callbacks = []
  if checkpoint_path is not None:
    checkpointer = ModelCheckpoint(checkpoint_path, monitor="val_acc", mode="max", save_best_only=True, save_weights_only=True)
    callbacks.append(checkpointer)
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

  model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
  model.fit_generator(
    batch_generator(db, batch_size=BATCH_SIZE,partition='train'), 
    validation_data=batch_generator(db, batch_size=BATCH_SIZE, partition='val'),
    steps_per_epoch=math.floor(238459 / BATCH_SIZE),
    validation_steps=math.floor(51129 / BATCH_SIZE),
    epochs=num_epoch,
    verbose=1,
    callbacks=callbacks
  )

  model.save(model_path)

def batch_generator(db, batch_size=100, partition='train'):
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
    yield (batch_x, batch_y)


def test():
  model = load_model(DEFAULT_MODEL_PATH)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Baseline NN')
  parser.add_argument('-p', '--model_path', dest='model_path')
  parser.add_argument('-e', '--num_epoch', dest='num_epoch', default=DEFAULT_NUM_EPOCH, type=int)
  parser.add_argument('-c', '--checkpoint', dest='checkpoint_path')
  args = parser.parse_args()
  train(**vars(args))
