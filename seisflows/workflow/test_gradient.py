import system
import optimize
import preprocess
import postprocess

from os.path import join
from seisflows.tools.config import SeisflowsParameters, SeisflowsPaths, ParameterError
from seisflows.tools.code import exists
from seisflows.tools import unix

PAR = SeisflowsParameters()
PATH = SeisflowsPaths()


class test_gradient(object):
    """ Generates synthetic data.
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

    def main(self):
        """ Generates data
        """

        # clean directories
        self.clean_directory(PATH.OUTPUT)
        self.clean_directory(PATH.SCRATCH)

        preprocess.setup()
        postprocess.setup()
        optimize.setup()

        print('Generating data...')
        system.run('solver', 'setup',
                   hosts='all')

        print('Computing gradient...')
        system.run('solver', 'compute_gradient',
                    hosts='all')
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