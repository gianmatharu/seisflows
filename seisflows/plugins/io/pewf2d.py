
from os.path import join
import numpy as np


def mread(path, parameters, prefix='', suffix=''):
    """ Multiparameter read, callable by a single mpi process
    """
    keys = []
    vals = []
    for key in sorted(parameters):
        val = read(path, key, prefix, suffix)
        keys += [key]
        vals += [val]

    return dict(zip(keys, vals))


def mwrite(model, path, prefix='', suffix=''):
    """ Multiparameter write. Writes a dictionary.
    """
    if model:
        for key in model.keys():
            write(model[key], path, key, prefix, suffix)


def read(path, parameter, prefix='', suffix=''):
    filename = prefix + parameter + suffix + '.bin'
    return _read(join(path, filename))


def write(v, path, parameter, prefix='', suffix=''):
    filename = prefix + parameter + suffix + '.bin'
    return _write(v, join(path, filename))


def _read(filename):
    """ Reads a single binary file
    """
    try:
        v = np.fromfile(filename, dtype='float32')
    except:
        raise IOError('Could not read file: {}'.format(filename))

    return v


def _write(v, filename):
    """ Writes a single binary file
    """
    v = np.asarray(v)
    try:
        v.astype('float32').tofile(filename)
    except:
        raise IOError('Could not write file: {}'.format(filename))
