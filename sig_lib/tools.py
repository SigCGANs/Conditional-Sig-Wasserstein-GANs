import itertools
import pickle

import numpy as np
import torch


class ExperimentGrid:
    """
    This is a highly convenient object to design experiments.
    """

    def __init__(self, root):
        self.grids = list()
        self.root = root

    def add_parameters(self, **kwargs):
        if len(self.grids) == 0:
            self.new_grid()
        grid = self.grids[-1]
        for k, val in kwargs.items():
            if not isinstance(val, list):
                val = [val]
            grid[k] = val
        pass

    def new_grid(self):
        self.grids.append(dict())

    def size(self):
        count = 0
        for grid in self.grids:
            values = list(grid.values())
            for parameters in itertools.product(*values):
                count += 1
        return count

    def iterate_over_grids(self):
        # need to think of a smarter way than this
        for grid in self.grids:
            values = list(grid.values())
            keys = list(grid.keys())
            for parameters in itertools.product(*values):
                hyperparameter_set = dict()
                experiment_directory = list()
                for key, param in zip(keys, parameters):
                    hyperparameter_set[key] = param
                    experiment_directory.append('%s=%s' % (key, param))
                experiment_directory = self.root + '_'.join(experiment_directory)
                yield hyperparameter_set, experiment_directory


def sample_indices(dataset_size, batch_size):
    indices = torch.from_numpy(np.random.choice(dataset_size, size=batch_size, replace=False)).cuda()
    # functions torch.-multinomial and torch.-choice are extremely slow -> back to numpy
    return indices


def pickle_it(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
