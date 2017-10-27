
import sys
import numpy as np

from seisflows.tools import unix
from seisflows.config import ParameterError, custom_import
from seisflows.plugins import optimize

PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']


class p_newton(custom_import('optimize', 'newton')):
    """ Implements Newton-CG algorithm
    """

    def compute_direction(self):
        self.restarted = False

        m = self.load('m_new')
        g = self.load('g_new')

        self.LCG.initialize()

        # loop over LCG iterations
        for self.ilcg in range(1, PAR.LCGMAX+1):
            if PAR.VERBOSE:
                print " LCG iteration", self.ilcg

            dm = self.load('LCG/p')

            # finite difference pertubation
            h = PAR.EPSILON/(max(abs(dm)))

            # compute Hessian-vector product by finite differences
            Hdm = self.apply_hessian(m, dm, h)
            self.save('Hdm', Hdm)

            # perform LCG iteration
            status = self.LCG.update(Hdm)

            if status > 0:
                # finalize model update
                dm = self.load('LCG/x')
                if self.dot(g,dm) >= 0:
                    print ' Newton direction rejected [not descent direction]'
                    dm = -g
                    self.restarted = True
                else:
                    self.LCG.finalize()
                self.save('p_new', dm)
                break


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
        return (self.load('g_lcg') - self.load('g_new'))/h


    def restart(self):
        # not required for this subclass since restarts are handled by 
        # compute_direction
        pass


    def apply_hess(self, path=''):
        """ Computes action of Hessian on a given model vector.
        """
        system = sys.modules['seisflows_system']

        system.run_single('solver', 'apply_hess')
        system.run('solver', 'prepare_apply_hess')
        system.run_single('solver', 'apply_hess', adjoint=True)