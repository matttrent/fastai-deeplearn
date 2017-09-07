#!/usr/bin/env python

from __future__ import division, print_function

import os
import sys
import click
import subprocess
import pandas as pd

from fastai import config, utils, kaggle

@click.command()
@click.option('-r', '--run', default=-1, help='run to test')
@click.option('-c', '--competition', help='kaggle competition name')
@click.argument('dataset')
def submit(run, competition, dataset):
    kaggle.submit(run, competition, dataset)


if __name__ == '__main__':
    submit()
