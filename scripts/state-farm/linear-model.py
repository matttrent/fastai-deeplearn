from fastai.state_farm import *

import numpy as np


t_batches = get_train_batches()
v_batches = get_valid_batches()



def get_lin_model():

    # starting with BatchNormalization saves us from having to normalize our 
    # input manually
    model = Sequential([
        BatchNormalization(axis=1, input_shape=(224, 224, 3)),
        Flatten(),
        Dense(10, activation='softmax')
    ])
    model.compile(
        Adam(), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model




lm = get_lin_model()
lm.summary()

rates = [
    (1e-5, 2),
    (1e-4, 4)
]

fastai.utils.fit_generator(lm, t_batches, rates, val_batches=v_batches)

# seeing consistency between repeated runs to evaluate sample size

r_batches = get_batches(get_data_path()+'valid', batch_size=2*BATCH_SIZE)
val_res = [
    lm.evaluate_generator(r_batches, r_batches.samples//r_batches.batch_size) 
    for i in range(10)
]
print(np.round(val_res, 2))
