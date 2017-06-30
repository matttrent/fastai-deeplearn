#!/usr/bin/env python

import sys
import click

import keras.models
from keras.preprocessing import image
import pandas as pd

from fastai import config, utils, fautils, kaggle


def submission_df(batches, preds):
    # construct dataframe of the submission
    index = pd.Series(
        [int(f[8:f.find('.')]) for f in batches.filenames],
        name='id'
    )
    df = pd.DataFrame({
        'label': preds[:, 1]
    },
        index=index
    )
    return df


@click.command()
@click.option('-r', '--run', default=1, help='run to test')
@click.option('-b', '--batch-size', default=64, help='size of batches')
@click.option('-c', '--competition', default=None, help='kaggle competition name')
@click.argument('dataset')
def train(run, batch_size, competition, dataset):

    # data set paths
    dset = config.DataSet(dataset)

    # load model
    model = keras.models.load_model(
        dset.path_for_run(run) + 'model.h5'
    )

    # load test batches
    from fastai.vgg16 import Vgg16
    test_batches = Vgg16.get_batches(
        dset.test_path, shuffle=False, batch_size=batch_size * 2,
        class_mode=None)

    # predict
    preds = model.predict_generator(test_batches, test_batches.nb_sample)

    # format dataframe
    df = submission_df(test_batches, preds)
    df.label = df.label.clip(0.05, 0.95)
    df.to_csv(dset.path_for_run(run) + 'submission.csv', index=True)

    # submit
    if competition is None:
        return
    kaggle.submit(run, competition, dataset)


if __name__ == '__main__':
    train()
