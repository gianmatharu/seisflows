import numpy as np
from seisflows.tools import sampling

class IndexedSourcePosition2D(object):
    """ Store 2D position coordinates.
    """
    def __init__(self, index, x, z):
        self.index = int(index)
        self.x = x
        self.z = z

    def print_pos(self):
        print('Index {:03d} at [{}, {}]'.format(self.index, self.x, self.z))


class SourceArray(object):
    """ Hold source positions
    """
    def __init__(self, sources=None):
        self.source_array = []
        if isinstance(sources, IndexedSourcePosition2D):
            self.source_array = [sources]
        if sources:
            self.source_array.extend(sources)

    def __getitem__(self, index):
        return self.source_array.__getitem__(index)

    def __len__(self):
        return len(self.source_array)

    @classmethod
    def fromfile(cls, file):
        arr = np.loadtxt(file)
        n = arr.shape[0]
        out = cls()

        for i in range(n):
            out.source_array += [IndexedSourcePosition2D(arr[i, 0], arr[i, 1], arr[i, 2])]

        return out

    def print_positions(self):
        print('Full list of sources...')
        for item in self.source_array:
            print('Source {:03d} at [{}, {}]'.format(item.index, item.x, item.z))
        print('\n')


# subsampling functions
def subsample(source_array, nsubset, scheme='uniform', p=None):
    """ Perform subsampling on source array
    """
    if not isinstance(source_array, SourceArray):
        raise TypeError('Input should be a SourceArray object.')

    f = getattr(sampling, scheme)
    index_list = f(len(source_array), nsubset, p=p)

    return SourceArray([source_array[i] for i in index_list])



