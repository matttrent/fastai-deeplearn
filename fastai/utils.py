from __future__ import print_function, division

import errno
import os
import glob
import shutil

import numpy as np
import pandas as pd

import keras.callbacks


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def copyfiles(src, dst, n):
    classes = glob.glob(src + '/*')
    for cls in classes:
        files = glob.glob(cls + '/*.jpg')
        shuf = np.random.permutation(files)
        for i in range(n):
            new_path = dst + '/' + os.path.join( *shuf[i].split('/')[2:] )
            # print 'cp', shuf[i], new_path
            shutil.copyfile(shuf[i], new_path)


def list_rate_schedule(lrates, output=True):
    sched = []
    last_lr = [0]

    for lr, n in lrates:
        sched += [lr] * n

    def lr_sched(epoch):
        lr = sched[-1]
        if epoch < len(sched):
            lr = sched[epoch]
        if output and lr != last_lr[0]:
            print('Learning rate: {}'.format(lr))
            last_lr[0] = lr
        return lr

    return lr_sched


def fit_generator(model, trn_batches, rates=None, val_batches=None, 
    callbacks=None, **kwargs):

    if val_batches is not None:
        kwargs['validation_data'] = val_batches
        kwargs['nb_val_samples'] = val_batches.nb_sample

    kwargs['nb_epoch'] = sum([p[1] for p in rates])

    kwargs['callbacks'] = []
    if callbacks is not None:
        kwargs['callbacks'] += callbacks

    if rates is not None:
        lrsched = keras.callbacks.LearningRateScheduler(
            list_rate_schedule(rates))
        kwargs['callbacks'].append(lrsched)

    return model.fit_generator(
        trn_batches,
        trn_batches.nb_sample,
        **kwargs
    )

