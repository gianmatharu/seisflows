
import sys
import numpy as np

from os.path import exists
from obspy.core import Stream, Trace

from seisflows.plugins import adjoint, misfit
from seisflows.tools import unix
from seisflows.config import ParameterError, custom_import

PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']

system = sys.modules['seisflows_system']


class saga(custom_import('preprocess', 'base')):
    """ SAGA data processing class

      Adds finite sum data misfit functions to base class
    """
    def sum_residuals(self, files, n=1):
        """ Sums squares of residuals (normalized by 1/N)

          INPUT
            FILES - files containing residuals
        """
        total_misfit = 0.
        for file in files:
            total_misfit += np.sum(np.loadtxt(file)**2.)
        return total_misfit/n