
from os.path import join

import numpy as np

from seisflows.tools import unix
from seisflows.tools.tools import loadtxt, savetxt
#from seisflows.tools.io import OutputWriter

from seisflows.plugins import preconds
from seisflows.plugins.optimize.LBFGS import LBFGS
from seisflows.plugins.optimize.LCG import LCG
 
 
class LBFGS_(LBFGS):
    """ Adapts L-BFGS from nonlinear optimization to preconditioning
    """
    pass


class PLCG(LCG):
    """ Preconditioned truncated-Newton CG solver

      Adds preconditioning and adaptive stopping to LCG base class
    """
    def __init__(self, path, eta=1., **kwargs):
        self.eta = eta

        super(PLCG, self).__init__(path, **kwargs)

        # prepare output writer
        self.logpath = join(path, 'output.plcg')
        #self.writer = OutputWriter(self.logpath, width=14)


    def apply_precond(self, r):
        if not self.precond:
            return r
        elif self.precond in dir(preconds):
            return self.precond(r)

        elif self.precond in ['LBFGS_3']:
            if self.iter == 1:
                self.LBFGS = LBFGS(self.path, memory=3)
                y = r
            elif self.ilcg == 0:
                S, Y = self.LBFGS.update()
                y = -self.LBFGS.apply(self.load('LCG/r'), S, Y)
            else:
                y = -self.LBFGS.apply(self.load('LCG/r'))
            return y

        elif self.precond in ['LBFGS_6']:
            if self.iter == 1:
                self.LBFGS = LBFGS(self.path, memory=6)
                y = r
            elif self.ilcg == 0:
                S, Y = self.LBFGS.update()
                y = -self.LBFGS.apply(self.load('LCG/r'), S, Y)
            else:
                y = -self.LBFGS.apply(self.load('LCG/r'))
            return y

        elif self.precond in ['LBFGS_9']:
            if self.iter == 1:
                self.LBFGS = LBFGS(self.path, memory=9)
                y = r
            elif self.ilcg == 0:
                S, Y = self.LBFGS.update()
                y = -self.LBFGS.apply(self.load('LCG/r'), S, Y)
            else:
                y = -self.LBFGS.apply(self.load('LCG/r'))
            return y

        else:
            raise ValueError



    def check_status(self, ap, verbose=True):
        """ Checks Eisenstat-Walker termination status
        """
        g0 = self.load('g_new')
        #g1 = self.load('LCG/r')
        g1 = g0 + ap

        LHS = _norm(g1)
        RHS = _norm(g0)

        if verbose:
            print ' ETA:', self.eta
            print ' RATIO:', LHS/RHS
            print ''

        #self.writer(
        #    self.iter,
        #    self.ilcg,
        #    LHS,
        #    RHS,
        #    LHS/RHS,
        #    eta1996)

        # check termination condition
        if LHS < self.eta * RHS:
            return _done
        else:
            return not _done

    def finalize(self, verbose=True):
        """ Update the forcing term in Eisenstat-Walker condition
            using condition 3.
            eta = a * (norm(g_new)/norm(g_old))**b
            a - [0, 1]
            b - [1, 2]
        """
        # for comparison, calculates forcing term proposed by
        # Eisenstat & Walker 1996
        try:
            a1 = 1
            a2 = 0.5*(1 + np.sqrt(5))

            g_new = _norm(self.load('g_new'))
            g_old = _norm(self.load('g_old'))

            # eta update
            eta_k = a1*(g_new/g_old)**a2

            # eta safeguard
            eta_s = a1*self.eta**a2

            # safeguard
            if eta_s > 0.1:
                if verbose:
                    print ' ETA trial:', eta_k
                    print ' ETA safeguard:', eta_s
                    print ''
                eta_k = max(eta_s, eta_k)

            if eta_k < 1:
                self.eta = eta_k

        except IOError:
            pass


### utility functions

_done = 0

def _norm(v):
    return float(np.linalg.norm(v))
