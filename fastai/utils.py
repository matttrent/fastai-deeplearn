import errno
import os

import pandas as pd


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def prediction_df(batches, preds):
    # construct dataframe of the submission
    index = pd.Series(
        [int(f[8:f.find('.')]) for f in batches.filenames],
        name='id'
    )
    df = pd.DataFrame({
        'label': preds[:, 1]
    },
        index=index
    )
    return df