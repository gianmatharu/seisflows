from glob import glob
from os.path import join
import sys
import numpy as np

from seisflows.seistools.ewf2d import iter_dirname, event_dirname
from seisflows.tools import msg
from seisflows.tools import unix
from seisflows.tools.array import loadnpy, savenpy
from seisflows.tools.code import divides, exists
from seisflows.tools.config import SeisflowsParameters, SeisflowsPaths, \
    ParameterError, custom_import

PAR = SeisflowsParameters()
PATH = SeisflowsPaths()

import system
import solver
import optimize
import preprocess
import postprocess


class wginversion(custom_import('workflow', 'pinversion')):
    """ Specialized Seismic inversion class for westgrid systems.

      Compute iterative non-linear inversion. Designed to fit PEWF2D solver.
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

        super(wginversion, self).check()

        if not (PAR.SYSTEM == 'westgrid' or PAR.SYSTEM == 'westgrid_pbs'):
            raise ValueError('wginversion can only be run with westgrid system class')


    def setup(self):
        """ Lays groundwork for inversion
        """
        # clean scratch directories
        if PAR.BEGIN == 1:
            unix.rm(PATH.SCRATCH)
            unix.mkdir(PATH.SCRATCH)

            preprocess.setup()
            postprocess.setup()
            optimize.setup()

        # initialize directories
        system.run('solver', 'setup',
                   hosts='all')

        # copy/generate data
        if PATH.DATA:
            print('Copying data...')
        else:
            print('Generating data...')
            system.run('solver', 'generate_data',
                        hosts='mpi_c')


    def compute_gradient(self):
        """ Compute gradients. Designed to avoid excessive storage
            of boundary files.
        """

        # output for inversion history
        unix.mkdir(join(PATH.OUTPUT, iter_dirname(optimize.iter)))

        print('Generating synthetics...')
        system.run('solver', 'generate_synthetics',
                    mode=1,
                    hosts='mpi_c')

        print('Prepare adjoint sources...')
        system.run('solver', 'prepare_eval_grad',
                   hosts='all')

        print('Computing gradient...')
        system.run('solver', 'compute_gradient',
                    hosts='mpi_c')

        postprocess.write_gradient(PATH.GRAD)

        # evaluate misfit function
        self.sum_residuals(path=PATH.SOLVER, suffix='new')


    def evaluate_function(self):
        """ Performs forward simulation to evaluate objective function
        """
        self.write_model(path=PATH.FUNC, suffix='try')

        system.run('solver', 'evaluate_function',
                   hosts='mpi_c')
        system.run('solver', 'process_trial_step',
                   hosts='all')

        self.sum_residuals(path=PATH.FUNC, suffix='try')


