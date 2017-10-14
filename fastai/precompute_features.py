from __future__ import print_function, division

import sys
import argparse

import fastai
from fastai.fautils import *

full_data_path = os.path.expanduser('~/data/state-farm/')
samp_data_path = os.path.expanduser('~/data/sample-state-farm/')
data_path = full_data_path
batch_size = 64

parser = argparse.ArgumentParser()
parser.add_argument(
    'mode', help='whether to cache or load features')

args = parser.parse_args()


# -----------------------------------------------------------------------------
# create batches
# -----------------------------------------------------------------------------

gen_t = image.ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.05,
    shear_range=0.1,
    rotation_range=15,
    channel_shift_range=20
)

t_batches = get_batches(
    data_path + 'train', batch_size=batch_size, shuffle=False)
v_batches = get_batches(
    data_path + 'valid', batch_size=2*batch_size, shuffle=False)
a_batches = get_batches(
    data_path + 'train', gen_t, batch_size=batch_size)

(
    val_classes, trn_classes, 
    val_labels, trn_labels, 
    val_filenames, filenames,
    test_filename
) = get_classes(data_path)


# -----------------------------------------------------------------------------
# construct initial VGG network, then split
# -----------------------------------------------------------------------------

if args.mode == 'store':

    vgg = vgg_ft(10)
    model = vgg.model

    lr = 1e-3
    epochs = 25.
    decay_rate = lr / epochs

    model.compile(
        optimizer=Adam(lr=1e-3, decay=decay_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    layers = model.layers

    last_conv_idx = [
        index 
        for index, layer in enumerate(layers)
        if type(layer) is Convolution2D
    ][-1]

    print(last_conv_idx)
    layers[last_conv_idx]

    conv_layers = layers[:last_conv_idx+1]
    conv_model = Sequential(conv_layers)

    fc_layers = layers[last_conv_idx+1:]
    fc_model = Sequential(fc_layers)

    # precompute, store

    t_features = conv_model.predict_generator(t_batches, t_batches.nb_sample)
    v_features = conv_model.predict_generator(v_batches, v_batches.nb_sample)

    save_array(data_path + 'train_convlayer_features.bc', t_features)
    save_array(data_path + 'valid_convlayer_features.bc', v_features)

    sys.exit()


if args.mode == 'load':

    # reload

    t_features = load_array(data_path + 'train_convlayer_features.bc')
    v_features = load_array(data_path + 'valid_convlayer_features.bc')

    print(t_features.shape)

    # create second network and train

    lr = 1e-5
    epochs = 25.
    decay_rate = lr / epochs

    model = Sequential([
        MaxPooling2D((2, 2), strides=(2, 2), input_shape=(512, 14, 14)),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(.5),
        Dense(4096, activation='relu'),
        Dropout(.5),
        Dense(10, activation='softmax'),
    ])

    model.compile(
        optimizer=Adam(lr=lr, decay=decay_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    h = model.fit(
        t_features, 
        trn_labels, 
        nb_epoch=25,
        batch_size=batch_size,
        validation_data=(v_features, val_labels)
    )

    sys.exit()
