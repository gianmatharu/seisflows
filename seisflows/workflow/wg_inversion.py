
import sys
from os.path import join

from seisflows.tools import unix
from seisflows.tools.tools import exists
from seisflows.tools.array import savenpy
from seisflows.config import ParameterError, custom_import
from seisflows.plugins.solver.pewf2d import iter_dirname

PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']

system = sys.modules['seisflows_system']
solver = sys.modules['seisflows_solver']
optimize = sys.modules['seisflows_optimize']
preprocess = sys.modules['seisflows_preprocess']
postprocess = sys.modules['seisflows_postprocess']


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
            setattr(PATH, 'MODELS', join(PATH.WORKDIR, 'models'))

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
        system.run('solver', 'setup')

        # copy/generate data
        if PATH.DATA:
            print('Copying data...')
        else:
            print('Generating data...')
            system.run_parallel('solver', 'generate_data')


    def compute_gradient(self):
        """ Compute gradients. Designed to avoid excessive storage
            of boundary files.
        """

        # output for inversion history
        unix.mkdir(join(PATH.OUTPUT, iter_dirname(optimize.iter)))

        print('Generating synthetics...')
        system.run_parallel('solver', 'generate_synthetics', mode=1)

        print('Prepare adjoint sources...')
        system.run('solver', 'prepare_eval_grad')

        print('Computing gradient...')
        system.run_parallel('solver', 'compute_gradient')

        postprocess.write_gradient(PATH.GRAD)
        dst = join(PATH.OPTIMIZE, 'g_new')
        savenpy(dst, solver.merge(solver.load(PATH.GRAD, suffix='_kernel')))

        # evaluate misfit function
        self.sum_residuals(path=PATH.SOLVER, suffix='new')


    def evaluate_function(self):
        """ Performs forward simulation to evaluate objective function
        """
        unix.rm(PATH.FUNC)
        unix.mkdir(PATH.FUNC)

        self.write_model(path=PATH.FUNC, suffix='try')

        system.run_parallel('solver', 'evaluate_function')
        system.run('solver', 'process_trial_step')
        self.sum_residuals(path=PATH.FUNC, suffix='try')


