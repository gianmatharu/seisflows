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
        Encoding holds a parameter for encoding. Currently
        it holds a time shift or a random number. 
    """
    def __init__(self):
        self.source_group = []
        self.encoding = []

    def __getitem__(self, index):
        return self.source_group.__getitem__(index)

    def group_sources(self, source_array, ngroups, repeat=False):
        """ Group sources uniformly with offsets.
        """
        if not isinstance(source_array, SourceArray):
            raise TypeError('Expected SourceArray object')

        # performs equidistant grouping
        for i in xrange(ngroups):
            if repeat:
                self.source_group += [SourceArray(source_array[:])]
            else:
                self.source_group += [SourceArray(source_array[i::ngroups])]

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

def random_encoder(n):
    """ Generate random phase encoding 
    """
    return np.random.choice([-1, 1], n)
    #return np.ones(n)


def plane_wave_encoder(p, x0, source_array):
    """ Generate plane wave time shifts for origin x0 and 
        ray parameter p
    """
    if not isinstance(source_array, SourceArray):
        raise TypeError('Expected SourceArray object.')

    shifts = map(lambda x: abs(p)*(x.x - x0), source_array)

    # reverse shifts if negative p
    if p < 0:
        shifts = shifts[::-1]

    return shifts


def shift_encoder(n, dt, max_dt):
    """ Generate random time shifts between a maximum shift
    """
    max_dt = abs(max_dt)
    dist = np.arange(-max_dt, max_dt+dt, dt)
    return np.random.choice(dist, n)


def generate_ray_parameters(pmin, pmax, n):

    ray_parameters = (pmax-pmin) * np.random.random_sample(n) + pmin
    ray_parameters *= np.random.choice([-1, 1], n)

    return ray_parameters


def decimate_source_array(source_array, ndecimate, random=False, batch=False):
    """ Return a decimated source array  
    """
    if not isinstance(source_array, SourceArray):
        raise TypeError('Input should be a SourceArray object.')

    n = len(source_array)

    if random:
        # perform batch-random sampling
        if batch:
            index_list = _random_batch_choice(n, ndecimate)
        else:
            index_list = np.random.choice(n, ndecimate, replace=False)
        index_list.sort()
    else:
        # perform uniform decimation (assumes ordered array).
        index_list = np.linspace(0, n, ndecimate, endpoint=False)
        index_list = [int(item) for item in index_list]

    return SourceArray([source_array[i] for i in index_list])

def _random_batch_choice(n, nchoice):
    """ Performs random sampling for batches. Assumes ordered array. 
        Warning, assumes ordered reflection geometry. 
    """
    index_list = []
    ninterval = int(n / nchoice)

    for i in xrange(nchoice):
        print np.arange(i*ninterval, (i+1)*ninterval)
        index_list.append(np.random.choice(np.arange(i*ninterval, (i+1)*ninterval)))

    return index_list

