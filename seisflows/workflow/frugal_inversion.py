from glob import glob
from os.path import join
import sys
import numpy as np

from seisflows.seistools.ewf2d import iter_dirname, event_dirname
from seisflows.tools import unix
from seisflows.tools.code import divides
from seisflows.tools.config import SeisflowsParameters, SeisflowsPaths, \
    ParameterError, custom_import

PAR = SeisflowsParameters()
PATH = SeisflowsPaths()

import system
import solver
import optimize
import preprocess
import postprocess


class frugal_inversion(custom_import('workflow', 'alt_inversion')):
    """ Seismic inversion base class.

      Compute iterative non-linear inversion. Designed to fit EWF2D solver.
      Follows a more generic design to base class.

      To allow customization, the inversion workflow is divided into generic 
      methods such as 'initialize', 'finalize', 'evaluate_function', 
      'evaluate_gradient', which can be easily overloaded.

      Calls to forward and adjoint solvers are abstracted through the 'solver'
      interface so that various forward modeling packages can be used
      interchangeably.

      Commands for running in serial or parallel on a workstation or cluster
      are abstracted through the 'system' interface.
    """

    def check(self):
        """ Check parameters and paths
        """

        super(frugal_inversion, self).check()

        # check PAR
        if 'NPROCMAX' not in PAR:
            raise ParameterError(PAR, 'NPROCMAX')

        if 'SAVERESIDUALS' not in PAR:
            setattr(PAR, 'SAVERESIDUALS', 1)


    def compute_gradient(self):
        """ Compute gradients. Designed to avoid excessive storage
            of boundary files.
        """

        # output for inversion history
        unix.mkdir(join(PATH.OUTPUT, iter_dirname(optimize.iter)))

        # compute gradients
        system.run('solver', 'compute_gradient',
                   hosts='all')
        postprocess.write_gradient(PATH.GRAD)

        # evaluate misfit function
        self.sum_residuals(path=PATH.SOLVER, suffix='main', set='all')
        self.sum_residuals(path=PATH.SOLVER, suffix='new', set='subset')


    def evaluate_function(self):
        """ Performs forward simulation to evaluate objective function
        """
        self.write_model(path=PATH.FUNC, suffix='try')

        system.run('solver', 'evaluate_function',
                   hosts='subset',
                   path=PATH.FUNC)

        self.sum_residuals(path=PATH.FUNC, suffix='try', set='subset')


    def sum_residuals(self, path='', suffix='', set='all'):
        """ Returns sum of squares of residuals for a subset of the data.
        """
        dst = PATH.OPTIMIZE +'/'+ 'f_' + suffix
        residuals = []

        if set == 'all':
            queue = range(PAR.NTASK)
        elif set == 'subset':
            queue = np.linspace(0, PAR.NTASK, PAR.NPROCMAX, dtype='int')
        else:
            raise KeyError('Accepted set values are all or subset')

        for itask in queue:
            src = path +'/'+ event_dirname(itask + 1) +'/'+ 'residuals'
            fromfile = np.loadtxt(src)
            residuals.append(fromfile**2.)
        np.savetxt(dst, [np.sum(residuals)])


    def save_residuals(self):
        src = join(PATH.OPTIMIZE, 'f_main')
        dst = join(PATH.OUTPUT, iter_dirname(optimize.iter), 'misfit')
        unix.mv(src, dst)