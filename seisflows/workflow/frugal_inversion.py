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

      Specialized class that reduces the number of events used to evaluate
      trial misfit functions. Ideally used with mpi_queue.

      Class requires that 3 extra routines be defined in the system class
      - system.getsubnode - must set PAR.LS_NODE = 'getsubnode'
      - system.run_subset
      - system.queue_subset

      Will perform trial misfit calculations for NPROCMAX events (assumed to
      be in a line). This reduces the number of forward calculations required
      to evaluate a line search.
    """

    def check(self):
        """ Check parameters and paths
        """

        super(frugal_inversion, self).check()

        # check PAR
        if 'NPROCMAX' not in PAR:
            raise ParameterError(PAR, 'NPROCMAX')

        # check for getsubnode
        if PAR.LS_NODE != 'getsubnode':
            raise ValueError('LS_NODE must be set to getsubnode in PAR')
        else:
            try:
                getattr(system, 'getsubnode')
            except:
                raise AttributeError('No method getsubnode found in system class')

        # check for run_subset
        if not getattr(system, 'run_subset', False):
            raise AttributeError('No run_subset method defined in system class')

        #check for queue_subset
        if not getattr(system, 'queue_subset', False):
            raise AttributeError('No queue_subset method defined in system class')

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

        system.run_subset('solver', 'evaluate_function',
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
            queue = system.queue_subset()
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