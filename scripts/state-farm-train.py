#!/usr/bin/env python

import os
import sys
import click
import yaml

import pandas as pd

from fastai import config, utils


def test_df(batches, preds):
    # construct dataframe of the submission
    index = pd.Series(
        [f for f in batches.filenames],
        name='id'
    )
    df = pd.DataFrame({
        'label': preds[:, 1]
    },
        index=index
    )
    return df.sort_index()


@click.command()
@click.option('-e', '--epochs', default=3, help='number of epochs')
@click.option('-b', '--batch-size', default=64, help='size of batches')
@click.option('--lr', '--learning-rate', default=0.01, help='learning rate')
@click.option('-t', '--num-trainable', default=1, help='train last n layers')
@click.option('-d', '--dropout', default=0.5, help='dropout')
@click.argument('dataset')
def train(epochs, batch_size, learning_rate, num_trainable, dropout, dataset):

    # data set paths
    dset = config.DataSet(dataset)

    utils.mkdir_p(dset.run_path)

    with open(dset.run_path + 'params.yaml', 'w') as f:
        f.write(yaml.dump({
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'trainable': num_trainable,
            'dropout': dropout,
            'dataset': dataset,
        }))

    # create model
    from keras import backend as K
    from keras.callbacks import CSVLogger
    from fastai.vgg16 import Vgg16
    vgg = Vgg16(dropout=dropout)

    # get the batches
    batches = vgg.get_batches(dset.train_path, batch_size=batch_size)
    val_batches = vgg.get_batches(dset.validate_path, batch_size=batch_size * 2)

    # fine tune the network and optimization
    vgg.finetune(batches)
    K.set_value(vgg.model.optimizer.lr, learning_rate)

    # fit the data
    csv_logger = CSVLogger(dset.run_path + 'train_log.csv', append=True)
    vgg.fit(batches, val_batches, nb_epoch=1, callbacks=[csv_logger])

    if epochs > 1:
        vgg.trainable_layers(num_trainable)
        vgg.fit(batches, val_batches, nb_epoch=epochs-1, callbacks=[csv_logger])

    model_fn = 'model.h5'
    vgg.model.save(dset.run_path + model_fn)

    # predict validation set and save
    batches, preds = vgg.test(dset.validate_path, batch_size=batch_size * 2)
    df = test_df(batches, preds)
    df.to_csv(dset.run_path + 'validate.csv', index=True)


if __name__ == '__main__':
    train()
