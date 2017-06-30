import os
import subprocess

from fastai import config


def submit(run, competition, dataset):

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

    subprocess.call(cmd)
