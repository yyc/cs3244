# CS3244 (2018) Group 25
Component Regularization for Domain-Specific Image Classification

## Set up & Quick Start

### Baseline

## Overview

Experiment1 Baseline model - `baseline.py`

Experiment2a Shi Yuan's - `not yet created`

Experiment2b Yu Chuan's - `not yet created`

Dataset (contains images and their classification)
* `data.h5`, *<partition> can be 'train', 'val', or 'test'*
    * **key**: 'ims_<partition>', **value**: array, i-th row contains i-th image of shape (3, 256, 256)
    * **key**: 'classes_<partition>', **value**: array, i-th row contains category (as int) of i-th recipe
    * **key**: 'impos_<partition>, **value**: array, i-th row contains list of image ids for the i-th recipe
    *Note! image ids returned by this array are 1 more than the true image id*

    * Others:
    * **key**: 'ids_<partition>', **value**: array, i-th row contains string id of i-th recipe


Helper package (/helper)
* `classes.py` to extract category labels from `classes1M.pkl`
* `image.py` to show image, has example code to show all images of a particular category

*Others*
* `mnist.py`, digits classifier created following a tutorial
* `resnet50.py`, `extract_titles.py` unused?


## Details


**If you are not too familiar with keras**, you can try out the tutorial in
https://elitedatascience.com/keras-tutorial-deep-learning-in-python
which is done once in `mnist.py`. Some of the method calls are different
due to changes in the keras api

We are trying to create 3 NN models:
1. Baseline model, normal image classifier
    - ResNet50 (an existing image classifier) to a fully connected layer
    - Analyze its performance
    - Refer to `baseline.py`
2. Shi Yuan's
3. Yu Chuan's

### Dataset
hdf5 file which is like a key-value database containing images, classes etc.
