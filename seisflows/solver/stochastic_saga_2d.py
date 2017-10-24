import sys
from os.path import join

import numpy as np

from seisflows.tools.seismic import getpar

from seisflows.config import ParameterError, custom_import

PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']

system = sys.modules['seisflows_system']


class stochastic_saga_2d(custom_import('solver', 'stochastic_saga'), custom_import('solver', 'specfem2d')):
    """ Adds stochastic optimization (SAGA) machinery to SPECFEM2D
    """

    def get_source_positions(self):
        """ Read in source positions.
            Order coincides with self._source_names.
        """
        positions = np.zeros((len(self._source_names), 2))
        for isrc, source_name in enumerate(self._source_names):
            source_file = join(PATH.SPECFEM_DATA, self.source_prefix + '_' + source_name)
            xs = getpar('xs', source_file, cast=float)
            zs = getpar('zs', source_file, cast=float)
            positions[isrc, :] = [xs, zs]

        return positions