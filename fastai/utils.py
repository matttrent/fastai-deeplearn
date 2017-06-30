import errno
import os
import glob
import shutil

import numpy as np
import pandas as pd



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
            new_path =  dst + '/' + os.path.join( *shuf[i].split('/')[2:] )
            # print 'cp', shuf[i], new_path
            shutil.copyfile(shuf[i], new_path)
