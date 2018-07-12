import numpy as np
from seisflows.plugins.misc.model import insert_gaussian

def volume(v, perc=None):
    """ Apply unit or percentage perturbation
    """
    v = np.asarray(v)

    if perc:
        val = (perc/100.) * abs(v).max()
    else:
        val = 1
    v += val

    return v, val


def check_spike_pars(parameters, size):
    """ Check spike input
    """
    if len(parameters) != 3:
        raise ValueError('Values missing in spike parameter')

    start, end, interval = parameters

    if start > end:
        raise ValueError('spike start greater than end')

    if start < 0 or start >= size:
        raise ValueError('spike start out of bounds')

    if end < 0 or end > size:
        raise ValueError('spike end out of bounds')

    if interval <= 0 or interval > size:
        raise ValueError('spike interval must be an integer < size')

    return


def get_spike_array(parameters):
    """ Return spike array
    """
    start, end, interval = parameters

    start = int(start)
    end = int(end)
    interval = int(interval)

    if start == end:
        return np.asarray([start])
    else:
        return np.arange(start, end, interval)


def spike(v, dims, xpos, zpos, perc):
    """ Apply spike perturbation to model
    """
    # reshape
    nx, nz = dims
    v = v.reshape((nz, nx))

    if perc:
        val = (perc/100.) * abs(v).max()
    else:
        val = 1

    for i in zpos:
        for j in xpos:
            v[i][j] += val

    v = v.reshape((nx * nz))

    return v, val

def get_spike_perturbation(v, dims, xpos, zpos, perc):
    """ Apply spike perturbation to model
    """
    # reshape
    nx, nz = dims
    v = v.reshape((nz, nx))
    dv = np.zeros(v.shape)

    if perc:
        val = (perc/100.) * abs(v).max()
    else:
        val = 1

    for i in zpos:
        for j in xpos:
            dv[i][j] += val

    dv = dv.reshape((nx * nz))

    return dv, val

def random(v, perc):
    """ Add zero-mean perturbation
    """
    if perc:
        val = (perc/100.) * abs(v).max()
    else:
        val = 1

    pert = val * np.random.randn(len(v))
    v += pert

    return v, pert


def gaussian(v, dims, xpos, zpos, perc, sigma):
    # reshape
    nx, nz = dims
    v = v.reshape((nz, nx))

    val = (perc/100.) * abs(v).max()

    # insert perturbations
    for iz in zpos:
        for ix in xpos:
            v = insert_gaussian(v, (iz, ix), sigma, val)

    v = v.reshape((nx * nz))

    return v, val
