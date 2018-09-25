import numpy as np
import lmdb
import os
import pickle
import h5py

np.random.seed(123)

from keras.models import Sequential, Model
from keras.layers import Dense
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical

resnet = ResNet50(input_shape=(3,256,256), pooling='max', classes=1024, include_top=False, weights=None)
top_model = Sequential()
top_model.add(Dense(1024, activation='softmax'))
model = Model(inputs=resnet.input, outputs=top_model(resnet.output))

data_path = 'data/'
keys_path = 'data/test_keys.pkl'
images_path = 'data/images/test'
classes_path = 'data/classes1M.pkl'
db_path = 'data/data.h5'

classes_file = open(classes_path, 'rb')
id2class = pickle.load(classes_file)
ind2class = pickle.load(classes_file)

db = h5py.File(db_path, 'r')

images = []
categories = []
for i in range(10): #actually len(f['ids_train'])
  id = db['ids_test'][i]
  category = db['classes_train'][i]
  for j in range(db['numims_train'][i]):
    index = db['impos_train'][i][j]
    image = db['ims_train'][index]
    images.append(image)
    categories.append(category)

train_x = np.array(images)
train_y = to_categorical(categories, 1024)

model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
model.fit(train_x, train_y, verbose=1)
