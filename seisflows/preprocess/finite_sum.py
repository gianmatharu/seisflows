
import sys
from os.path import join
from functools import wraps

import numpy as np

from seisflows.plugins import adjoint, misfit, readers, writers
from seisflows.config import ParameterError, custom_import
from seisflows.tools.math import Normalize


PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']


class finite_sum(custom_import('preprocess', 'pewf2d')):
    """ Finite sum preprocessing class.
        Used for stochastic optimization.
    """
    def setup(self):
        """ Sets up data preprocessing machinery
        """
        # define misfit function and adjoint trace generator
        if PAR.MISFIT:
            self.misfit = getattr(misfit, PAR.MISFIT)
            self.adjoint = Normalize(getattr(adjoint, PAR.MISFIT),
                                     PAR.NSOURCES)
        elif PAR.BACKPROJECT:
            self.adjoint = getattr(adjoint, PAR.BACKPROJECT)

        # define seismic data reader and writer
        self.reader = getattr(readers, PAR.FORMAT)
        self.writer = getattr(writers, PAR.FORMAT)

    def sum_residuals(self, files):
        """ Sums squares of residuals

          INPUT
            FILES - files containing residuals
        """
        total_misfit = super(finite_sum, self).sum_residuals(files)

        return (0.5/PAR.NSOURCES) * total_misfit
