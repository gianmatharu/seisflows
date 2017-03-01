
from os.path import join
import numpy as np


def mread(path, parameters, suffix=''):
    """ Multiparameter read, callable by a single mpi process
    """
    keys = []
    vals = []
    for key in sorted(parameters):
        val = read(path, key, suffix=suffix)
        keys += [key]
        vals += [val]
    return keys, vals


def mwrite(path, model):
    """ Multiparameter write. Writes a dictionary.
    """
    if (model):
        for key in model.keys():
            write(path, key, model[key])


def read(path, parameter, suffix=''):
    """ Reads a single binary file
    """
    filename = join(path, '{}{}.bin'.format(parameter, suffix))
    return np.fromfile(filename, dtype='float32')


def write(path, parameter, v):
    """ Writes a single binary file
    """
    v = np.asarray(v)

    filename = join(path, '{}.bin'.format(parameter))
    v.astype('float32').tofile(filename)
