import numpy as np
import lmdb
import os
import pickle

np.random.seed(123)

from keras.models import Sequential
from keras.layers import Dense
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image


model = ResNet50(weights='imagenet')

data_path = 'data/'
keys_path = 'data/test_keys.pkl'
images_path = 'data/images/test'

env = lmdb.open(os.path.join(data_path, 'test_lmdb'), max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)

ids = pickle.load(open(keys_path, 'rb'))

def path_for(filename):
  return os.path.join(images_path, filename[0], filename[1], filename[2], filename[3], filename)

for id in ids[:10]:
  with env.begin(write=False) as txn:
    byte_sample = txn.get(id.encode())
  sample = pickle.loads(byte_sample, encoding='latin')
  for image_file in sample['imgs']:
    path = path_for(image_file['id'])
    if not os.path.exists(path):
      print("Missing: {} | {}".format(id, path))
      continue
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    print('Prediction: {} | {}'.format(path, decode_predictions(preds, top=3)[0]))

