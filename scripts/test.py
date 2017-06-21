#!/usr/bin/env python

import glob
import sys

import click

from fastai import config, utils


@click.command()
@click.option('-b', '--batch-size', default=64, help='size of batches')
@click.option('-r', '--run', default=-1, help='run to test')
@click.argument('dataset')
def train(batch_size, run, dataset):
    click.echo('training')
    click.echo(sys.argv)

    # data set paths
    dataset = config.DataSet(dataset)

    click.echo(dataset.train_path)
    utils.mkdir_p(dataset.train_path)

    click.echo(dataset.validate_path)
    utils.mkdir_p(dataset.validate_path)

    if run == -1:
        run = None
    run_path = dataset.path_for_run(run)
    weights = sorted(glob.glob(run_path + 'ft*.h5'))
    if len(weights) == 0:
        raise RuntimeError('no saved weights in {}'.format(run_path))
    latest_weights = weights[-1]

    # create model
    from fastai.vgg16 import Vgg16
    vgg = Vgg16()

    # get the batches
    batches = vgg.get_batches(dataset.train_path, batch_size=batch_size)
    # fine tune the network and optimization
    vgg.finetune(batches)
    # load pre-trained weights
    vgg.model.load_weights(latest_weights)

    # get test patches, predictions and filenames
    batches, preds = vgg.test(dataset.test_path, batch_size=batch_size * 2)

    df = utils.submission_df(batches, preds)
    df.label = df.label.clip(min=0.05, max=0.95)
    df.to_csv(run_path + 'submission.csv', index=True)

    from fastai import fautils
    fautils.save_array(run_path + 'test_preds.dat', preds)
    fautils.save_array(run_path + 'filenames.dat', batches.filenames)


if __name__ == '__main__':
    train()
