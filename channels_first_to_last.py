import h5py
import numpy as np

def build_flipped_dataset(db, new_db, partition):
    ids = db['ids_{}'.format(partition)]
    classes = db['classes_{}'.format(partition)]
    impos = db['impos_{}'.format(partition)]
    ims = db['ims_{}'.format(partition)]
    numims = db['numims_{}'.format(partition)]

    new_ims = [np.rollaxis(img, 0, 3) for img in ims]

    print(new_ims[0].shape)

    new_db.create_dataset('ids_{}'.format(partition), data=ids)
    new_db.create_dataset('classes_{}'.format(partition), data=classes)
    new_db.create_dataset('numims_{}'.format(partition), data=numims)
    new_db.create_dataset('impos_{}'.format(partition), data=impos)

    new_db.create_dataset('ims_{}'.format(partition), data=new_ims)



db_path = 'data/subset_train_val.h5'
new_db_path = 'data/flipped_subset.h5'

db = h5py.File(db_path, 'r')
new_db = h5py.File(new_db_path, 'w')

build_flipped_dataset(db, new_db, "train")
build_flipped_dataset(db, new_db, "val")

new_db.flush()
new_db.close()
db.close()
