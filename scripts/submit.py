#!/usr/bin/env python

import os
import sys
import click
import subprocess
import pandas as pd

from fastai import config, utils

@click.command()
@click.option('-r', '--run', default=-1, help='run to test')
@click.option('-c', '--competition', help='kaggle competition name')
@click.argument('dataset')
def submit(run, competition, dataset):
    print('submit')
    print(sys.argv)

    if competition is None:
        competition = dataset

    # data set paths
    dataset_obj = config.DataSet(dataset)

    if run == -1:
        run = None
    run_path = dataset_obj.path_for_run(run)

    cmd = [
        'kg',
        'submit',
        '-u', os.environ['KAGGLE_USERNAME'],
        '-p', os.environ['KAGGLE_PASSWORD'],
        '-c', competition,
        run_path + 'submission.csv'
    ]

    print(' '.join(cmd))
    subprocess.call(cmd)

    # preds = fautils.load_array(dataset.path_for_run() + 'test_preds.dat')
    # filenames = fautils.load_array(dataset.path_for_run() + 'filenames.dat')
    #
    # index = pd.Series(
    #     [int(f[8:f.find('.')]) for f in filenames],
    #     name='id'
    # )
    #
    # df = pd.DataFrame({
    #     'label': preds[:,1]
    #     },
    #     index=index
    # )
    #
    # df.to_csv(dataset.path_for_run() + 'submission.csv', index=True)


if __name__ == '__main__':
    submit()
