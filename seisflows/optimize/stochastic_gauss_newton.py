
import sys
import numpy as np

from seisflows.config import ParameterError, custom_import

PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']


class stochastic_gauss_newton(custom_import('optimize', 'stochastic_newton')):
    """ Implements subsampled Gauss-Newton CG algorithm
    """

    def hessian_product(self, h):
        # Gauss-Newton Hessian-vector product using second-order adjoints.
        return self.load('g_lcg')/h


    def apply_hess(self, path=''):
        """ Computes action of Hessian on a given model vector.
        """
        system = sys.modules['seisflows_system']
        solver = sys.modules['seisflows_solver']


        system.run_single('solver', 'apply_hess',
                          model_dir=PATH.HESS+'/model')
        solver.prepare_apply_hess()
        system.run_single('solver', 'apply_hess',
                          model_dir=PATH.MODEL_EST,
                          adjoint=True)