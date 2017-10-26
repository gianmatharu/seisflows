import sys

from os.path import join
from seisflows.tools import unix
from seisflows.tools.tools import exists
from seisflows.tools.array import savenpy
from seisflows.config import ParameterError
from seisflows.workflow.base import base


PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']

system = sys.modules['seisflows_system']
solver = sys.modules['seisflows_solver']
optimize = sys.modules['seisflows_optimize']
preprocess = sys.modules['seisflows_preprocess']
postprocess = sys.modules['seisflows_postprocess']


class test_gradient(base):
    """ Test gradient computation.
    """

    def check(self):
        """ Checks parameters and paths
        """
        # check paths
        if 'DATA' not in PATH:
            setattr(PATH, 'DATA', None)

        if 'GRAD' not in PATH:
            setattr(PATH, 'GRAD', join(PATH.SCRATCH, 'evalgrad'))

        if 'OPTIMIZE' not in PATH:
            setattr(PATH, 'OPTIMIZE', join(PATH.SCRATCH, 'optimize'))

        if 'MODEL_TRUE' not in PATH:
            raise ParameterError(PATH, 'MODEL_TRUE')

        if 'MODELS' not in PATH:
            setattr(PATH, 'MODELS', join(PATH.SUBMIT, 'models'))

        if 'MODEL_EST' not in PATH:
            setattr(PATH, 'MODEL_EST', join(PATH.MODELS, 'model_est'))

        if 'STORE' not in PATH:
            setattr(PATH, 'STORE', join(PATH.WORKDIR, 'gradients'))

        # check parameters
        if not PAR.USE_STF_FILE:
            raise ValueError('Must use stf for gradient calculations.')
        else:
            if not exists(join(PATH.SOLVER_INPUT, PAR.STF_FILE)):
                raise IOError('Source time function file not found.')

        if PAR.SYSTEM != 'serial':
            raise ValueError('Use system class "serial" here.')

        if PAR.SOLVER != 'pewf2d':
            raise ValueError('Use solver class "pewf2d" here.')


    def main(self):
        """ Compute gradient
        """

        # clean directories
        self.clean_directory(PATH.OUTPUT)
        self.clean_directory(PATH.SCRATCH)

        preprocess.setup()
        postprocess.setup()
        optimize.setup()

        self.generate_data()
        self.evaluate_gradient()
        self.store_gradient()
        print('Finished\n')

    def generate_data(self):

        system.run('solver', 'setup')

        if PATH.DATA:
            print('Copying data')
        else:
            print('Generating data...')
            system.run_single('solver', 'generate_data')

    def evaluate_gradient(self):

        print('Generating synthetics...')
        system.run_single('solver', 'generate_synthetics',
                          mode=1)

        print('Prepare adjoint sources...')
        system.run('solver', 'prepare_eval_grad')

        print('Computing gradient...')
        system.run_single('solver', 'compute_gradient')

        postprocess.write_gradient(PATH.GRAD)

        src = PATH.GRAD
        dst = join(PATH.OPTIMIZE, 'g_new')
        savenpy(dst, solver.merge(solver.load(src, suffix='_kernel')))


    def store_gradient(self):
        """ Store individual gradients
        """
        self.clean_directory(PATH.STORE)
        system.run_single('solver', 'export_gradient',
                          path=PATH.STORE)

    def clean_directory(self, path):
        """ If dir exists clean otherwise make
        """

        if not exists(path):
            unix.mkdir(path)
        else:
            unix.rm(path)
            unix.mkdir(path)