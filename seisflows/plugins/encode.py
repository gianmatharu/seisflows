import numpy as np


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


class SourceGroups(object):
    """ Group sources. Each data member is a list that contains a
        'supershot'. Each supershot contains N sources.
    """
    def __init__(self):
        self.source_group = []
        self.encoding = []

    def __getitem__(self, index):
        return self.source_group.__getitem__(index)

    def group_sources(self, source_array, ngroups):
        """ Group sources uniformly with offsets.
        """
        if not isinstance(source_array, SourceArray):
            raise TypeError('Expected SourceArray object')

        # performs equidistant grouping
        for i in range(ngroups):
            self.source_group += [SourceArray(source_array[i::ngroups])]

    def generate_encoding(self, reset=False):
        """ Store source encoding
        """
        if reset:
            self.encoding = []

        if self.source_group:
            for group in self.source_group:
                self.encoding += [generate_random_plus_minus(len(group))]

    def print_groups(self):
        """ Print source groups
        """
        for i, group in enumerate(self.source_group):
            print('Group {:03d}'.format(i))

            for j, source in enumerate(group):
                if self.encoding:
                   print('Source {:03d} at [{}, {}], Encoding: {}'.format(
                         source.index, source.x, source.z, self.encoding[i][j]))
                else:
                   print('Source {:03d} at [{}, {}]'.format(
                       source.index, source.x, source.z))

            print('\n')


# auxiliary functions

def generate_random_plus_minus(n):

    #return np.ones(n)
    return np.random.choice([-1, 1], n)
