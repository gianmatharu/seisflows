
import numpy as np


def uniform(n, nsubset, p=None):
    """ Randomly subsample source array
    """
    return np.sort(np.random.choice(n, nsubset, replace=False))


def non_uniform(n, nsubset, p=None):
    """ Randomly subsample source array with non-uniform distribution
    """
    return np.sort(np.random.choice(n, nsubset, p=p, replace=False))


def pseudo_random(n, nsubset, p=None):
    """ Perform pseudo-random subsampling on source array
    """
    return np.sort(_random_batch_choice(n, nsubset))


def decimate(n, nsubset, p=None):
    """ Perform regular shot decimation
    """
    return range(n)[::(n/nsubset)]


def _random_batch_choice(n, nsubset):
    """ Performs random sampling for batches. Assumes ordered array.
        Warning, assumes ordered reflection geometry.
    """
    index_list = []
    ninterval = int(n / nsubset)

    for i in xrange(nsubset):
        index_list.append(np.random.choice(np.arange(i*ninterval, (i+1)*ninterval)))

    return index_list
