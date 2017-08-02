#!/usr/bin/env python

# IMPORTANT: needs to be called before keras is imported
from fastai.config import setup_dl
setup_dl()

import os
import click
import yaml

import pandas as pd

from fastai import config, utils

from keras.callbacks import CSVLogger


def results_path(dataset):
    run_num = int(os.environ.get('DOMINO_RUN_NUMBER'))
    run_str = '{}-{:03}/'.format(dataset, run_num)

    path = os.path.expandvars(os.environ.get('RESULTS_ROOT_PATH'))
    path = os.path.join(path, run_str)
    return path


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
@click.argument('dataset')
def train(epochs, batch_size, learning_rate, dataset):

    dset = config.DataSet(dataset)
    dset.domino_helper()

    run_path = results_path(dataset)
    utils.mkdir_p(run_path)

    with open(run_path + 'params.yaml', 'w') as f:
        f.write(yaml.dump({
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'dataset': dataset
        }))

    # config keras
    # config.setup_dl()
    from keras import backend as K
    print(K.backend(), K.image_dim_ordering(), K.epsilon(), K.floatx())

    # K.set_image_dim_ordering('th')
    # K.set_epsilon(1e-7)
    # K.set_floatx('float32')
    # print(K.backend(),K.image_dim_ordering(), K.epsilon(), K.floatx())

    # create model
    from fastai.vgg16 import Vgg16
    vgg = Vgg16()

    # get the batches
    batches = vgg.get_batches(dset.train_path, batch_size=batch_size)
    val_batches = vgg.get_batches(dset.validate_path, batch_size=batch_size * 2)

    # fine tune the network and optimization
    vgg.finetune(batches)
    K.set_value(vgg.model.optimizer.lr, learning_rate)

    # fit the data
    csv_logger = CSVLogger(run_path + 'train_log.csv')
    vgg.fit(batches, val_batches, nb_epoch=epochs, callbacks=[csv_logger])

    # save the model, unless it's a sample
    if 'sample' not in dataset:
        model_fn = 'model.h5'
        vgg.model.save(run_path + model_fn)

    # predict validation set and save
    batches, preds = vgg.test(dset.validate_path, batch_size=batch_size * 2)
    df = test_df(batches, preds)
    df.to_csv(run_path + 'validate.csv', index=True)


if __name__ == '__main__':
    train()
