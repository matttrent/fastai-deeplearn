import os
import glob
import shutil
import tarfile

from fastai import utils

DATASET_TAR_PATH = os.environ.get(
    'DOMINO_MATTTRENT_KAGGLE_DOGSCATS_WORKING_DIR')
DATASET_PATH = os.path.expandvars('$DOMINO_WORKING_DIR/tmp')


def setup_theano():
    destfile = os.path.expandvars('$HOME/.theanorc')
    srcfile  = os.path.expandvars('$DOMINO_WORKING_DIR/.theanorc')
    open(destfile, 'a').close()
    shutil.copyfile(srcfile, destfile)

    print('Finished setting up Theano')


def setup_keras():
    srcpath = os.path.expandvars('$DOMINO_WORKING_DIR/keras')
    dstpath = os.path.expandvars('$HOME/.keras')

    os.symlink(srcpath, dstpath)

    print('Finished setting up Keras')


def setup_dl():
    # setup_theano()
    setup_keras()


class DataSet(object):

    def __init__(self, dataset):
        self._run_number = None
        self._dataset = dataset

    def domino_helper(self):
        print('dataset path: {}'.format(DATASET_PATH))
        print('dataset tarpath: {}'.format(DATASET_TAR_PATH))
        utils.mkdir_p(DATASET_PATH)
        dataset_tar = os.path.join(
            DATASET_TAR_PATH, '{}.tar.gz'.format(self._dataset))
        print('dataset tarfile: {}'.format(dataset_tar))
        tar = tarfile.open(dataset_tar)
        tar.extractall(DATASET_PATH)
        tar.close()

    def _path_helper(self, *args):
        path = os.path.join(DATASET_PATH, self._dataset)

        path = os.path.join(path, *args)
        if path[-1] != '/':
            path += '/'
        return path

    def _last_run(self):
        run_paths = glob.glob(
            os.path.join(self.results_path, 'run_*')
        )
        run_paths = sorted(run_paths)

        if len(run_paths) is 0:
            return None

        run_num = os.path.split(run_paths[-1])[-1][4:]
        return int(run_num)

    @property
    def train_path(self):
        return self._path_helper('train')

    @property
    def test_path(self):
        return self._path_helper('test')

    @property
    def validate_path(self):
        return self._path_helper('valid')

    @property
    def results_path(self):
        return self._path_helper('results')

    @property
    def run_number(self):
        if self._run_number is None:
            run_number = self._last_run()
            if run_number is None:
                self._run_number = 0
            else:
                self._run_number = run_number + 1

        return self._run_number

    @property
    def run_path(self):
        return self.path_for_run()

    def path_for_run(self, run_num=None):
        if run_num is None:
            run_num = self.run_number

        run_str = 'run_{:03d}'.format(run_num)
        return self._path_helper(self.results_path, run_str)