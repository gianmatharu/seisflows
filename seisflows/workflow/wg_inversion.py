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


class wg_inversion(custom_import('workflow', 'p_inversion')):
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

        super(wg_inversion, self).check()

        # check parameters
        if not PAR.USE_STF_FILE:
            raise ValueError('Must use stf for gradient calculations.')
        else:
            if not exists(join(PATH.SOLVER_INPUT, PAR.STF_FILE)):
                raise IOError('Source time function file not found.')

        if PAR.SYSTEM not in ['parallel', 'westgrid']:
            raise ValueError('wg_inversion can only be run with parallel or westgrid system class')

        # check paths
        if 'MODELS' not in PATH:
            setattr(PATH, 'MODELS', join(PATH.SUBMIT, 'models'))

        if 'MODEL_TRUE' not in PATH:
            raise ParameterError(PATH, 'MODEL_TRUE')

        if 'MODEL_EST' not in PATH:
            setattr(PATH, 'MODEL_EST', join(PATH.MODELS, 'model_est'))


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
        unix.rm(PATH.FUNC)
        unix.mkdir(PATH.FUNC)

        self.write_model(path=PATH.FUNC, suffix='try')

        system.run('solver', 'evaluate_function',
                   hosts='mpi_c')
        system.run('solver', 'process_trial_step',
                   hosts='all')

        self.sum_residuals(path=PATH.FUNC, suffix='try')


