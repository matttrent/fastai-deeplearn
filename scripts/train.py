#!/usr/bin/env python

import sys
import click

from fastai import config, utils, fautils

@click.command()
@click.option('-e', '--epochs', default=3, help='number of epochs')
@click.option('-b', '--batch-size', default=64, help='size of batches')
@click.option('--ft', '--finetune-layers', default=1,
              help='finetune N last layers')
@click.option('--lr', '--learning-rate', default=0.01, help='learning rate')
@click.argument('dataset')
def train(epochs, batch_size, finetune_layers, learning_rate, dataset):
    print('training')
    print(
        '{} {} {} {}'.format(epochs, batch_size, finetune_layers, learning_rate)
    )
    print(sys.argv)

    # data set paths
    dataset = config.DataSet(dataset)

    # click.echo(dataset.train_path)
    # click.echo(dataset.validate_path)
    # click.echo(dataset.run_path)
    utils.mkdir_p(dataset.train_path)
    utils.mkdir_p(dataset.validate_path)
    utils.mkdir_p(dataset.run_path)

    # create model
    from fastai.vgg16 import Vgg16
    vgg = Vgg16()

    # get the batches
    batches = vgg.get_batches(dataset.train_path, batch_size=batch_size)
    val_batches = vgg.get_batches(
        dataset.validate_path, batch_size=batch_size * 2)

    # fine tune the network and optimization
    vgg.finetune(batches, finetune_layers)
    vgg.model.optimizer.lr = learning_rate

    for epoch in range(epochs):
        print("Running epoch: {}".format(epoch))
        vgg.fit(batches, val_batches, nb_epoch=1)
        latest_weights_filename = 'ft%d.h5' % epoch
        vgg.model.save_weights(dataset.run_path + latest_weights_filename)

    print("Completed {} fit operations".format(epochs))

    batches, preds = vgg.test(dataset.test_path, batch_size=batch_size * 2)

    df = utils.submission_df(batches, preds)
    df.label = df.label.clip(0.05, 0.95)
    df.to_csv(dataset.run_path + 'submission.csv', index=True)


if __name__ == '__main__':
    train()
