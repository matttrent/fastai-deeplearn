from __future__ import print_function, division

import sys
import argparse

import fastai
import fastai.utils

from fastai.fautils import *

import pandas as pd


FULL_DATA_PATH = os.path.expanduser('~/data/state-farm/')
SAMP_DATA_PATH = os.path.expanduser('~/data/sample-state-farm/')
BATCH_SIZE = 64


# -----------------------------------------------------------------------------
# create batches
# -----------------------------------------------------------------------------

def get_data_path(sample=True):
    if sample:
        return SAMP_DATA_PATH
    return FULL_DATA_PATH


def get_train_batches(sample=True):
    data_path = FULL_DATA_PATH
    if sample:
        data_path = SAMP_DATA_PATH
    return get_batches(data_path + 'train', batch_size=BATCH_SIZE)


def get_valid_batches(sample=True):
    data_path = FULL_DATA_PATH
    if sample:
        data_path = SAMP_DATA_PATH
    return get_batches(
        data_path + 'valid', batch_size=2*BATCH_SIZE, shuffle=False)
    

# -----------------------------------------------------------------------------
# regularized linear model
# -----------------------------------------------------------------------------

def get_reglin_model():

    model = Sequential([
        BatchNormalization(axis=1, input_shape=(3, 224, 224)),
        Flatten(),
        Dense(10, activation='softmax', W_regularizer=l2(0.01))
    ])
    model.compile(
        Adam(), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model


# -----------------------------------------------------------------------------
# single dense layer
# -----------------------------------------------------------------------------

def get_fc_model():

    model = Sequential([
        BatchNormalization(axis=1, input_shape=(3, 224, 224)),
        Flatten(),
        Dense(100, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='softmax')
    ])
    model.compile(
        Adam(), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model


# -----------------------------------------------------------------------------
# simple conv layers
# -----------------------------------------------------------------------------

def get_conv_model(t_batches=None, v_batches=None, train=True):

    if t_batches is None:
        t_batches = get_train_batches()

    if v_batches is None:
        v_batches = get_valid_batches()

    model = Sequential([
        BatchNormalization(axis=1, input_shape=(3, 224, 224)),
        Convolution2D(32, 3, 3, activation='relu'),
        BatchNormalization(axis=1),
        MaxPooling2D((3, 3)),
        Convolution2D(64, 3, 3, activation='relu'),
        BatchNormalization(axis=1),
        MaxPooling2D((3, 3)),
        Flatten(),
        Dense(200, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='softmax')
    ])
    model.compile(
        Adam(), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    if not train:
        return model
    
    model.optimizer.lr.set_value(1e-4)
    h = model.fit_generator(
        t_batches, 
        t_batches.nb_sample, 
        nb_epoch=2, 
        validation_data=v_batches, 
        nb_val_samples=v_batches.nb_sample
    )

    model.optimizer.lr.set_value(1e-3)
    h = model.fit_generator(
        t_batches, 
        t_batches.nb_sample, 
        nb_epoch=4, 
        validation_data=v_batches, 
        nb_val_samples=v_batches.nb_sample
    )
    
    return model
