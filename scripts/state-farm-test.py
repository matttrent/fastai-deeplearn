#!/usr/bin/env python

import sys
import click

import keras.models
import pandas as pd

from fastai import config, utils, fautils


def submission_df(preds, test_batches, classes):
    # construct dataframe of the submission
    index = pd.Series(
        [f.split('/')[-1] for f in test_batches.filenames],
        name='img'
    )

    df = pd.DataFrame(
        preds,
        index=index,
        columns=classes
    )

    return df.sort_index()


@click.command()
@click.option('-r', '--run', default=-1, help='run to test')
@click.option('-b', '--batch-size', default=64, help='size of batches')
@click.argument('dataset')
def train(run, batch_size, dataset):
    # print('training')
    # print(
    #     '{} {} {}'.format(epochs, batch_size, learning_rate)
    # )
    # print(sys.argv)

    # data set paths
    dset = config.DataSet(dataset)

    # click.echo(dataset.train_path)
    # click.echo(dataset.validate_path)
    # click.echo(dataset.run_path)
    # utils.mkdir_p(dataset.train_path)
    # utils.mkdir_p(dataset.validate_path)
    # utils.mkdir_p(dataset.run_path)

    model = keras.models.load_model(
        dset.path_for_run(run) + 'model.h5'
    )

    # create model
    from fastai.vgg16 import Vgg16
    vgg = Vgg16()

    # get the batches
    batches = vgg.get_batches(dataset.train_path, batch_size=batch_size)
    val_batches = vgg.get_batches(
        dataset.validate_path, batch_size=batch_size * 2)

    # fine tune the network and optimization
    vgg.finetune(batches)
    vgg.model.optimizer.lr = learning_rate

    for epoch in range(epochs):
        print("Running epoch: {}".format(epoch))
        vgg.fit(batches, val_batches, nb_epoch=1)
        latest_weights_filename = 'ft%d.h5' % epoch
        vgg.model.save_weights(dataset.run_path + latest_weights_filename)

    print("Completed {} fit operations".format(epochs))

    test_batches, preds = vgg.test(dataset.test_path, batch_size=batch_size * 2)

    df = submission_df(preds, test_batches, vgg.classes)
    # df.label = df.label.clip(0.05, 0.95)
    df.to_csv(dataset.run_path + 'submission.csv', index=True)


if __name__ == '__main__':
    train()
