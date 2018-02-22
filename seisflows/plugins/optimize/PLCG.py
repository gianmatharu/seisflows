
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
    def __init__(self, path, eta=0.9, cond=1, **kwargs):
        self.eta = eta
        self.eta_init = eta
        self.cond = cond

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
        g1 = self.load('LCG/r')

        LHS = _norm(g1)
        RHS = _norm(g0)

        if verbose:
            print ' ETA:', self.eta
            print ' RATIO:', LHS/RHS
            print ' NORM RESIDUAL:', LHS
            print ''

        # check termination condition
        if LHS < self.eta * RHS:
            return _done
        else:
            return not _done

    def finalize(self, verbose=True):
        """ Update the forcing term in Eisenstat-Walker condition.
        """
        # Store previous residual
        unix.cp('LCG/r', 'LCG/r_old')

        # Calculates forcing term proposed by Eisenstat & Walker 1996
        if self.cond == 1:
            update_forcing_term = self.forcing_term_1
        elif self.cond == 2:
            update_forcing_term = self.forcing_term_2
        elif self.cond == 3:
            update_forcing_term = self.forcing_term_3
        else:
            raise ValueError('Cond must be one of 1, 2, or 3.')

        try:
            update_forcing_term(verbose=verbose)
        except IOError:
            print('Eta not updated')

    def forcing_term_1(self, verbose=True):
        """ Implements forcing term 1 in Eisenstat & Walker 1996
        """
        print 'Using 1'
        res = self.load('LCG/r_old')
        g_new = self.load('g_new')
        g_old = self.load('g_old')

        eisenvect = g_new - res

        # eta update and safeguard
        eta_k = _norm(eisenvect) / _norm(g_old)
        eta_s = self.eta**((1.+np.sqrt(5))/2.)

        self.eta = self._safeguard(eta_k, eta_s, verbose=verbose)

    def forcing_term_2(self, verbose=True):
        """ Implements forcing term 2 in Eisenstat & Walker 1996
        """
        print 'Using 2'

        res = self.load('LCG/r_old')
        g_new = self.load('g_new')
        g_old = self.load('g_old')

        # eta update and safeguard
        eta_k = abs(_norm(g_new) - _norm(res)) / _norm(g_old)
        eta_s = self.eta**((1.+np.sqrt(5))/2.)

        self.eta = self._safeguard(eta_k, eta_s, verbose=verbose)

    def forcing_term_3(self, verbose=True):
        """ Implements forcing term 3 in Eisenstat & Walker 1996.
        """
        print 'Using 3'

        a1 = 0.95
        a2 = 1

        g_new = _norm(self.load('g_new'))
        g_old = _norm(self.load('g_old'))

        # eta update and safeguard
        eta_k = a1*(g_new/g_old)**a2
        eta_s = a1*self.eta**a2

        self.eta = self._safeguard(eta_k, eta_s, verbose=verbose)

    def _safeguard(self, eta_k, eta_s, verbose=True):
        """ Forcing term safeguard. Takes max value
            between trial (eta_k) and safeguard (eta_s).
        """
        if eta_s > 0.1:
            if verbose:
                print ' ETA trial:', eta_k
                print ' ETA safeguard:', eta_s
                print ''
                eta_k = max(eta_s, eta_k)

        # avoid large eta
        if eta_k >= 1.0:
            eta_k = self.eta_init

        return eta_k

### utility functions

_done = 0

def _norm(v):
    return float(np.linalg.norm(v))

