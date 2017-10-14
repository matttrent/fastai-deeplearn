from __future__ import print_function, division

import sys
import argparse

import fastai
from fastai.fautils import *

import pandas as pd


full_data_path = os.path.expanduser('~/data/state-farm/')
samp_data_path = os.path.expanduser('~/data/sample-state-farm/')
data_path = samp_data_path
batch_size = 64


parser = argparse.ArgumentParser()
parser.add_argument(
    'model', help='which model to run')

args = parser.parse_args()


# -----------------------------------------------------------------------------
# create batches
# -----------------------------------------------------------------------------

t_batches = get_batches(data_path + 'train', batch_size=batch_size)
v_batches = get_batches(
    data_path + 'valid', batch_size=2*batch_size, shuffle=False)

(
    val_classes, trn_classes, 
    val_labels, trn_labels, 
    val_filenames, trn_filenames,
    tst_filenames
) = get_classes(data_path)


# -----------------------------------------------------------------------------
# linear model
# -----------------------------------------------------------------------------

def get_lin_model():

    # starting with BatchNormalization saves us from having to normalize our 
    # input manually
    model = Sequential([
        BatchNormalization(axis=1, input_shape=(3, 224, 224)),
        Flatten(),
        Dense(10, activation='softmax')
    ])
    model.compile(
        Adam(), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model


if args.model == 'lin-model':

    print('lin-model')

    lm = get_lin_model()
    lm.summary()

    lm.fit_generator(
        t_batches, 
        t_batches.nb_sample, 
        nb_epoch=1, 
        validation_data=v_batches, 
        nb_val_samples=v_batches.nb_sample
    )

    _ = np.round(lm.predict_generator(t_batches, t_batches.nb_sample)[:10], 2)
    print(_)

    # lower learning rate and trying again

    lm = get_lin_model()
    lm.optimizer.lr.set_value(1e-5)
    lm.fit_generator(
        t_batches, 
        t_batches.nb_sample, 
        nb_epoch=2, 
        validation_data=v_batches, 
        nb_val_samples=v_batches.nb_sample
    )
    lm.optimizer.lr.set_value(1e-3)
    lm.fit_generator(
        t_batches, 
        t_batches.nb_sample, 
        nb_epoch=4,
        validation_data=v_batches, 
        nb_val_samples=v_batches.nb_sample
    )

    # seeing consistency between repeated runs to evaluate sample size

    r_batches = get_batches(data_path+'valid', batch_size=2*batch_size)
    val_res = [
        lm.evaluate_generator(r_batches, r_batches.nb_sample) 
        for i in range(10)
    ]
    print(np.round(val_res, 2))

    sys.exit()


# -----------------------------------------------------------------------------
# regularized linear model
# -----------------------------------------------------------------------------

def get_reglin_model():

    model = Sequential([
        BatchNormalization(axis=1, input_shape=(3, 224, 224)),
        Flatten(),
        Dense(10, activation='softmax', W_regularizer=l2(0.01))
    ])
    model.compile(
        Adam(), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model


if args.model == 'reg-lin-model':

    print('reg-lin-model')

    rlm = get_reglin_model()
    rlm.optimizer.lr.set_value(1e-5)
    rlm.fit_generator(
        t_batches, 
        t_batches.nb_sample, 
        nb_epoch=2, 
        validation_data=v_batches, 
        nb_val_samples=v_batches.nb_sample
    )
    rlm.optimizer.lr.set_value(1e-4)
    rlm.fit_generator(
        t_batches, 
        t_batches.nb_sample, 
        nb_epoch=4, 
        validation_data=v_batches, 
        nb_val_samples=v_batches.nb_sample
    )

    sys.exit()


# -----------------------------------------------------------------------------
# single dense layer
# -----------------------------------------------------------------------------

def get_fc_model():

    model = Sequential([
        BatchNormalization(axis=1, input_shape=(3, 224, 224)),
        Flatten(),
        Dense(100, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='softmax')
    ])
    model.compile(
        Adam(), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model


if args.model == 'fc-model':

    fc = get_fc_model()
    fc.summary()

    fc.optimizer.lr.set_value(1e-5)
    fc.fit_generator(
        t_batches, 
        t_batches.nb_sample, 
        nb_epoch=2, 
        validation_data=v_batches, 
        nb_val_samples=v_batches.nb_sample
    )

    fc.optimizer.lr.set_value(0.01)
    fc.fit_generator(
        t_batches, 
        t_batches.nb_sample, 
        nb_epoch=5, 
        validation_data=v_batches, 
        nb_val_samples=v_batches.nb_sample
    )

    sys.exit()


# -----------------------------------------------------------------------------
# simple conv layers
# -----------------------------------------------------------------------------

def get_conv_model(t_batches=t_batches, v_batches=v_batches, train=True):

    model = Sequential([
        BatchNormalization(axis=1, input_shape=(3, 224, 224)),
        Convolution2D(32, 3, 3, activation='relu'),
        BatchNormalization(axis=1),
        MaxPooling2D((3, 3)),
        Convolution2D(64, 3, 3, activation='relu'),
        BatchNormalization(axis=1),
        MaxPooling2D((3, 3)),
        Flatten(),
        Dense(200, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='softmax')
    ])
    model.compile(
        Adam(), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    if not train:
        return model
    
    model.optimizer.lr.set_value(1e-4)
    h = model.fit_generator(
        t_batches, 
        t_batches.nb_sample, 
        nb_epoch=2, 
        validation_data=v_batches, 
        nb_val_samples=v_batches.nb_sample
    )

    model.optimizer.lr.set_value(1e-3)
    h = model.fit_generator(
        t_batches, 
        t_batches.nb_sample, 
        nb_epoch=4, 
        validation_data=v_batches, 
        nb_val_samples=v_batches.nb_sample
    )
    
    return model


if args.model == 'conv-model':

    cm = get_conv_model()
    cm.summary()

    sys.exit()


# -----------------------------------------------------------------------------
# data augmentation
# -----------------------------------------------------------------------------

if args.model == 'data-augment':

    gen_t = image.ImageDataGenerator(width_shift_range=0.1)
    batches = get_batches(data_path+'train', gen_t, batch_size=batch_size)
    model = get_conv_model(batches)

    gen_t = image.ImageDataGenerator(height_shift_range=0.05)
    batches = get_batches(data_path+'train', gen_t, batch_size=batch_size)
    model = get_conv_model(batches)

    gen_t = image.ImageDataGenerator(shear_range=0.1)
    batches = get_batches(data_path+'train', gen_t, batch_size=batch_size)
    model = get_conv_model(batches)

    gen_t = image.ImageDataGenerator(rotation_range=15)
    batches = get_batches(data_path+'train', gen_t, batch_size=batch_size)
    model = get_conv_model(batches)

    gen_t = image.ImageDataGenerator(channel_shift_range=20)
    batches = get_batches(data_path+'train', gen_t, batch_size=batch_size)
    model = get_conv_model(batches)

    sys.exit()


# -----------------------------------------------------------------------------
# all together
# -----------------------------------------------------------------------------

if args.model == 'all-together':

    gen_t = image.ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.05,
        shear_range=0.1,
        rotation_range=15,
        channel_shift_range=20
    )
    batches = get_batches(data_path + 'train', gen_t, batch_size=batch_size)

    model = get_conv_model(batches)

    model.optimizer.lr.set_value(0.0001)
    model.fit_generator(
        batches, batches.nb_sample, 
        nb_epoch=5, 
        validation_data=v_batches, nb_val_samples=v_batches.nb_sample)
    model.fit_generator(
        batches, batches.nb_sample, 
        nb_epoch=25, 
        validation_data=v_batches, nb_val_samples=v_batches.nb_sample)

    # TODO: what does this do?
    vf_batches = get_batches(
        os.path.expanduser('~/data/state-farm/') + 'valid', 
        batch_size=2*batch_size, shuffle=False)
    model.evaluate_generator(vf_batches, vf_batches.nb_sample)

    model.summary()

    model.save(data_path + 'state-farm-cnn.h5')

    sys.exit()


# -----------------------------------------------------------------------------
# full training plot
# -----------------------------------------------------------------------------

if args.model == 'full-train-plot':

    history = []

    gen_t = image.ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.05,
        shear_range=0.1,
        rotation_range=15,
        channel_shift_range=20
    )
    batches = get_batches(data_path + 'train', gen_t, batch_size=batch_size)

    model = get_conv_model(train=False)
    model.optimizer.lr.set_value(1e-4)
    h = model.fit_generator(
        batches, 
        batches.nb_sample, 
        nb_epoch=2, 
        validation_data=v_batches, 
        nb_val_samples=v_batches.nb_sample
    )
    history.append(h)

    model.optimizer.lr.set_value(1e-3)
    h = model.fit_generator(
        batches, 
        batches.nb_sample, 
        nb_epoch=4, 
        validation_data=v_batches, 
        nb_val_samples=v_batches.nb_sample
    )
    history.append(h)

    model.optimizer.lr.set_value(0.0001)
    h = model.fit_generator(
        batches, batches.nb_sample, 
        nb_epoch=5, 
        validation_data=v_batches, nb_val_samples=v_batches.nb_sample)
    history.append(h)
    h = model.fit_generator(
        batches, batches.nb_sample, 
        nb_epoch=25, 
        validation_data=v_batches, nb_val_samples=v_batches.nb_sample)
    history.append(h)

    acc = []
    val_acc = []
    for h in history:
        acc += h.history['acc']
        val_acc += h.history['val_acc']
        
    plt.plot(acc)
    plt.plot(val_acc)
    plt.show()

    sys.exit()


# -----------------------------------------------------------------------------
# reload model
# -----------------------------------------------------------------------------

if args.model == 'reload-model':

    model = keras.models.load_model(samp_data_path + 'state-farm-cnn.h5')

    t_batches = get_batches(full_data_path + 'train', batch_size=batch_size)
    v_batches = get_batches(
        full_data_path + 'valid', batch_size=2*batch_size, shuffle=False)

    (
        val_classes, trn_classes, 
        val_labels, trn_labels, 
        val_filenames, filenames,
        test_filename
    ) = get_classes(full_data_path)

    batches = get_batches(
        full_data_path + 'train', gen_t, batch_size=batch_size)

    h = model.fit_generator(
        batches, batches.nb_sample, 
        nb_epoch=5, 
        validation_data=v_batches, nb_val_samples=v_batches.nb_sample
    )

    sys.exit()
