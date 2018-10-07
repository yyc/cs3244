import lmdb
import os
import pickle
import h5py
import math


def batch_extractor(db, new_db, batch_size=100, partition='train'):
  ids = db['ids_{}'.format(partition)]
  classes = db['classes_{}'.format(partition)]
  impos = db['impos_{}'.format(partition)]
  ims = db['ims_{}'.format(partition)]
  numims = db['numims_{}'.format(partition)]
  new_db.create_dataset('ids_{}'.format(partition), data=ids[:batch_size])
  new_db.create_dataset('classes_{}'.format(partition), data=classes[:batch_size])
  new_db.create_dataset('numims_{}'.format(partition), data=numims[:batch_size])
  
  new_img_ids = []
  new_impos = []
  for ind in range(batch_size):
    img_ids = db['impos_{}'.format(partition)][ind]
    num_ims = numims[ind]
    img_idx = [img_ids[i] for i in range(num_ims)]
    new_ids = [len(new_img_ids) + i if i < num_ims else 0 for i in range(len(img_ids))]
    new_impos.append(new_ids)
    new_img_ids.extend(img_idx)

  new_db.create_dataset('impos_{}'.format(partition), data=new_impos)
  new_db.create_dataset('ims_{}'.format(partition), data=[ims[id] for id in new_img_ids])
  return new_db

  
db_path = 'data/data.h5'
new_db_path = 'data/sample_val_data.h5'

db = h5py.File(db_path, 'r')
new_db = h5py.File(new_db_path, 'w')

batch_extractor(db, new_db, batch_size=1000, partition="val")
new_db.flush()
