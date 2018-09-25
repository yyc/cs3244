import lmdb
import pickle
import os

env = lmdb.open(os.path.join('data', 'val_lmdb'), max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)

keys = pickle.load(open('data/val_keys.pkl', 'rb'))

for key in keys[:1]:
  with env.begin(write=False) as txn:
    byte_sample = txn.get(key.encode())
  data = pickle.loads(byte_sample, encoding="latin")
  print(list(data.keys())
