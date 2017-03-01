import sys

from os.path import join
from seisflows.tools import unix
from seisflows.tools.tools import exists
from seisflows.config import ParameterError

PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']

system = sys.modules['seisflows_system']
optimize = sys.modules['seisflows_optimize']
preprocess = sys.modules['seisflows_preprocess']
postprocess = sys.modules['seisflows_postprocess']


class test_gradient(object):
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
        """ Generates data
        """
        # clean directories
        self.clean_directory(PATH.OUTPUT)
        self.clean_directory(PATH.SCRATCH)

        preprocess.setup()
        postprocess.setup()
        optimize.setup()

        system.run('solver', 'setup',
                   hosts='all')

        if PATH.DATA:
            print('Copying data')
        else:
            print('Generating data...')
            system.run('solver', 'generate_data',
                        hosts='head')

        print('Generating synthetics...')
        system.run('solver', 'generate_synthetics',
                    mode=1,
                    hosts='head')

        print('Prepare adjoint sources...')
        system.run('solver', 'prepare_eval_grad',
                   hosts='all')

        print('Computing gradient...')
        system.run('solver', 'compute_gradient',
                    hosts='head')

        postprocess.write_gradient(PATH.GRAD)
        print('Finished')

    def clean_directory(self, path):
        """ If dir exists clean otherwise make
        """

        if not exists(path):
            unix.mkdir(path)
        else:
            unix.rm(path)
            unix.mkdir(path)