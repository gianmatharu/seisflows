
import sys
import numpy as np

from seisflows.tools import unix
from seisflows.config import ParameterError, custom_import
from seisflows.plugins import optimize

PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']


class stochastic_newton(custom_import('optimize', 'p_newton')):
    """ Implements subsampled Newton-CG algorithm
    """
    def check(self):
        """ Checks parameters and paths
        """
        super(stochastic_newton, self).check()

        if PAR.SOLVER != 'stochastic_newton':
            raise ValueError('Use "stochastic_newton" solver class')

        if PAR.PREPROCESS != 'finite_sum':
            raise ValueError('Use preprocessing class "finite_sum"')


    def compute_direction(self):
        solver = sys.modules['seisflows_solver']
        solver.select_sources()
        super(stochastic_newton, self).compute_direction()


    def hessian_product(self, h):
        """ Compute Hessian-vector product usign finite-difference.
        """
        self.postprocess_sub_gradient()
        return (self.load('g_lcg') - self.load('g_sub'))/h


    def apply_hess(self, path=''):
        """ Computes action of Hessian on a given model vector.
        """
        system = sys.modules['seisflows_system']
        solver = sys.modules['seisflows_solver']


        system.run_single('solver', 'apply_hess',
                          model_dir=PATH.HESS+'/model')
        solver.prepare_apply_hess()
        system.run_single('solver', 'apply_hess',
                          model_dir=PATH.HESS+'/model',
                          adjoint=True)


    def postprocess_sub_gradient(self):
        # apply postprocessing to sub gradient
        solver = sys.modules['seisflows_solver']

        # combine data over subset
        unix.mkdir(PATH.HESS +'/' + 'gradient_sub')
        solver.combine_subset(path=PATH.HESS+'/'+'gradient_sub')

        if PAR.SMOOTH > 0:
            solver.smooth(path=PATH.HESS+'/'+'gradient_sub',
                          span=PAR.SMOOTH)

        gradient = solver.load(PATH.HESS+'/'+'gradient_sub', suffix='_kernel')

        if PAR.RESCALE:
            gradient = solver.rescale.rescale_gradient(gradient)

        g = solver.merge(gradient)

        if PATH.MASK:
            # apply mask
            g *= solver.merge(solver.load(PATH.MASK))

        self.save('g_sub', g)
