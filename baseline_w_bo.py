import argparse
import math
import os

import GPy
import GPyOpt
import h5py
import numpy as np

# Use GPU 1 since other people are using 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
from keras.layers import Dense, Dropout
from keras.applications.resnet50 import ResNet50
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint

DEFAULT_MODEL_PATH = 'baseline_model.h5'


def new_model(nodes, layers, dropout):
    resnet = ResNet50(input_shape=(3, 256, 256), pooling='avg', include_top=False, weights='imagenet')
    # Freeze resnet layers
    for layer in resnet.layers:
        layer.trainable = False
    top_model = Sequential()
    for i in range(layers):
        top_model.add(Dense(nodes, activation='linear'))
        top_model.add(Dropout(dropout))
    top_model.add(Dense(1048, activation='softmax'))
    model = Model(inputs=resnet.input, outputs=top_model(resnet.output))
    return model


def open_model(filename):
    return load_model(filename)


def train_func_for_bo(model_path=None, checkpoint_path=None):
    def train(args):
        nodes, layers, batch_size, epochs, dropout = args[0]
        nodes, layers, batch_size, epochs = map(int, [nodes, layers, batch_size, epochs])
        # Set directory for model load and save:
        model = new_model(nodes, layers, dropout)
        print("initialized new model")
        callbacks = []
        if checkpoint_path is not None:
            checkpointer = ModelCheckpoint(checkpoint_path, monitor="val_acc", mode="max", save_best_only=True,
                                           save_weights_only=True)
            callbacks.append(checkpointer)

        # Set default directories
        db_path = 'data/data.h5'
        db = h5py.File(db_path, 'r')

        model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
        model.fit_generator(
            batch_generator(db, batch_size=batch_size, partition='train'),
            validation_data=batch_generator(db, batch_size=batch_size, partition='val'),
            steps_per_epoch=math.floor(238459 / batch_size),
            validation_steps=math.floor(51129 / batch_size),
            epochs=epochs,
            verbose=1,
            callbacks=callbacks
        )

        # Save exp model
        head, tail = os.path.split(model_path)
        exp_model_path = head + "_".join(map(lambda x: str(int(round(x))),
                                             [nodes, layers, batch_size, epochs, dropout * 10000])) + "_" + tail
        model.save(exp_model_path)
        print("Saved model to: " + exp_model_path)

        least_loss = min(model.history.history['val_loss'])
        print("Least Loss is: " + str(least_loss))
        return least_loss
    return train


def batch_generator(db, batch_size=100, partition='train'):
    ids = db['ids_{}'.format(partition)]
    classes = db['classes_{}'.format(partition)]
    impos = db['impos_{}'.format(partition)]
    ims = db['ims_{}'.format(partition)]
    numims = db['numims_{}'.format(partition)]
    partition_size = len(ids)
    while (True):
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


def train_w_bo(model_path=None, checkpoint_path=None, num_exp=1):
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    bounds = [{'name': 'nodes', 'type': 'discrete', 'domain': (64, 128, 256, 512, 1024, 2048)},
              {'name': 'layers', 'type': 'discrete', 'domain': (4, 8, 12, 16, 20, 24, 28, 32)},
              {'name': 'batch_size', 'type': 'discrete', 'domain': (10, 100, 250, 500)},
              {'name': 'epochs', 'type': 'discrete', 'domain': (40, 80, 120, 160, 200)},
              {'name': 'dropout', 'type': 'continuous', 'domain': (.0, .5)}]

    train = train_func_for_bo(model_path=model_path, checkpoint_path=checkpoint_path)

    bo = GPyOpt.methods.BayesianOptimization(f=train, domain=bounds,
                                             kernel=GPy.kern.Matern32(len(bounds), ARD=True),
                                             acquisition_type='EI', model_type='GP', ARD=True,
                                             initial_design_numdata=1)
    model_path = str(os.path.basename(model_path).split('.')[0]) + '_bomodel'
    for i in range(num_exp):
        print("\nRunning Exp No. " + str(i+1))
        bo.run_optimization(max_iter=1)
    bo.save_models(model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline NN w BO')
    parser.add_argument('-p', '--model_path', dest='model_path')
    parser.add_argument('-c', '--checkpoint', dest='checkpoint_path')
    parser.add_argument('--num_exp', dest='num_exp', type=int, default=3)
    args = parser.parse_args()
    train_w_bo(**vars(args))
