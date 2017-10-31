
import sys
from os.path import join

import numpy as np

from seisflows.config import ParameterError, custom_import


PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']


class finite_sum(custom_import('preprocess', 'pewf2d')):
    """ Finite sum preprocessing class.
        Used for stochastic optimization.
    """

    def sum_residuals(self, files):
        """ Sums squares of residuals

          INPUT
            FILES - files containing residuals
        """
        total_misfit = super(finite_sum, self).sum_residuals(files)

        return (1.0/PAR.NSOURCES) * total_misfit
