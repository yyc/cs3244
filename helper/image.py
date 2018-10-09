import numpy as np
from matplotlib import pyplot as plt

from helper.classes import extract_classes


def view_img(db, key, idx, channels_first=True):
    orig_img_arr = db[key][idx]
    chan_last_img_arr = np.moveaxis(orig_img_arr, 0, -1) if channels_first else orig_img_arr
    plt.imshow(chan_last_img_arr)
    plt.show()

if __name__ == '__main__':
    import h5py
    db_path = '../data/data.h5'
    db = h5py.File(db_path, 'r')
    numims = db['numims_val']
    classes = db['classes_val']
    impos = db['impos_val']
    class_labels = extract_classes('../data/classes1M.pkl')

    i = 312 # Change this value
    category = classes[i]
    category = category - 1
    print(category)
    for j in range(numims[i]):
        index = impos[i][j]
        view_img(db, 'ims_val', index - 1)
    print(class_labels[category])

