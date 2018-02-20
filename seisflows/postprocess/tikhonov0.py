import sys
import numpy as np

from seisflows.config import ParameterError, custom_import

PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']

system = sys.modules['seisflows_system']
solver = sys.modules['seisflows_solver']


class tikhonov0(custom_import('postprocess', 'regularize')):
    """ Adds regularization options to base class

        Available options include 0-, 1-, and 2- order Tikhonov and total
        variation regularization. While the underlying theory is classical,
        application to unstructured numerical grids via the
        "seisflows.tools.math.nabla" operator is somewhat complicated.

        So far, can only be used for 2D inversion, because the required spatial
        derivative operator "nabla" is not yet available for 3D grids.

        Tikhonov regularization is applied to ||m - m0|| where m0 is a prior
        model.
    """

    def check(self):
        """ Checks parameters and paths
        """
        super(tikhonov0, self).check()

    def sum_residuals(self, path):
        """ Evaluate regularization term
        """
        residuals = 0.
        m = solver.load(path, rescale=PAR.RESCALE)
        m0 = solver.load(PATH.MODEL_INIT, rescale=PAR.RESCALE)

        for key in solver.parameters:
            #print m[key].max(), m0[key].max()
            residuals += 0.5 * np.sum((m[key]-m0[key])**2)

        return residuals

    def nabla(self, m, key):
        """ Evaluate regularization gradient term
        """
        m0 = solver.load(PATH.MODEL_INIT, rescale=PAR.RESCALE)
        return m - m0[key]
