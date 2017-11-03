
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

    def compute_direction(self):
        solver = sys.modules['seisflows_solver']
        solver.select_sources()
        super(stochastic_newton, self).compute_direction()


    def apply_hessian(self, m, dm, h):
        """ Computes the action of the Hessian on a given vector through
          solver calls
        """
        solver = sys.modules['seisflows_solver']
        postprocess = sys.modules['seisflows_postprocess']

        self.save('m_lcg', m + h*dm)

        solver.save(solver.split(m + h*dm),
                PATH.HESS+'/'+'model')

        self.apply_hess(path=PATH.HESS)

        postprocess.write_gradient(path=PATH.HESS+'/'+'gradient',
                                   solver_path=PATH.HESS)

        self.save('g_lcg', solver.merge(solver.load(
                PATH.HESS+'/'+'gradient', suffix='_kernel')))

        unix.rm(PATH.HESS+'_debug')
        unix.mv(PATH.HESS, PATH.HESS+'_debug')
        unix.mkdir(PATH.HESS)

        #unix.rm(PATH.HESS)
        #unix.mkdir(PATH.HESS)

        return self.hessian_product(h)

    def hessian_product(self, h):
        # for Gauss-Newton model updates simply overload this method
        solver = sys.modules['seisflows_solver']

        # combine data over subset
        unix.mkdir(PATH.HESS +'/' + 'gradient_sub')
        solver.combine_subset(path=PATH.HESS+'/' +'gradient_sub')
        solver.smooth(path=PATH.HESS+'/' +'gradient_sub',
                      span=PAR.SMOOTH)
        self.save('g_sub', solver.merge(solver.load(
                PATH.HESS+'/'+'gradient_sub', suffix='_kernel')))

        if PATH.MASK:
            # apply mask
            g_sub = self.load('g_sub')
            g_sub *= solver.merge(solver.load(PATH.MASK))
            self.save('g_sub', g_sub)

        return (self.load('g_lcg') - self.load('g_sub'))/h


    def restart(self):
        # not required for this subclass since restarts are handled by
        # compute_direction
        pass

    def apply_hess(self, path=''):
        """ Computes action of Hessian on a given model vector.
        """
        system = sys.modules['seisflows_system']
        solver = sys.modules['seisflows_solver']


        system.run_single('solver', 'apply_hess')
        solver.prepare_apply_hess()
        system.run_single('solver', 'apply_hess', adjoint=True)