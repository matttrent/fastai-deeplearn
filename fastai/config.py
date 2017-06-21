import os
import glob

PROJECT_NAME = 'fastai-deep-learning'

DATA_PATH = os.path.join(os.path.expanduser('~/data'), PROJECT_NAME)


class DataSet(object):

    def __init__(self, dataset):
        self._run_number = None
        self._dataset = dataset

    def _path_helper(self, *args):
        path = os.path.join(DATA_PATH, self._dataset)

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