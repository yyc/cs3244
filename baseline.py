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

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Flatten, Dropout
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical

BATCH_SIZE = 20

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

def train():
#  model = open_model('models/baseline_model_dropout_imagenet.h5')
  model = new_model()
  data_path = 'data/'
  keys_path = 'data/test_keys.pkl'
  images_path = 'data/images/test'
  classes_path = 'data/classes1M.pkl'
  db_path = 'data/data.h5'

  classes_file = open(classes_path, 'rb')
  id2class = pickle.load(classes_file)
  ind2class = pickle.load(classes_file)

  db = h5py.File(db_path, 'r')

  model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
  model.fit_generator(
    batch_generator(db, batch_size=BATCH_SIZE,partition='train'), 
    validation_data=batch_generator(db, batch_size=BATCH_SIZE, partition='val'),
    steps_per_epoch=math.floor(238459 / BATCH_SIZE),
    validation_steps=math.floor(51129 / BATCH_SIZE),
    epochs=40, 
    verbose=1
  )

  model.save('models/baseline_correct_indices.h5')

def batch_generator(db, batch_size=100, partition='train'):
  ids = db['ids_{}'.format(partition)]
  classes = db['classes_{}'.format(partition)]
  impos = db['impos_{}'.format(partition)]
  ims = db['ims_{}'.format(partition)]
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
      for j in range(db['numims_train'][i]):
        index = db['impos_train'][i][j] - 1
        image = db['ims_train'][index]
        images.append(image)
        categories.append(category)

    batch_x = np.array(images)
    batch_y = to_categorical(categories, 1048)
    yield( batch_x, batch_y )


def test():
  model = load_model('baseline_model.h5')
  
train()
