
from glob import glob
from os.path import join

from seisflows.tools import unix
from seisflows.tools.code import exists
from seisflows.tools.config import SeisflowsParameters, SeisflowsPaths, \
    ParameterError, custom_import

PAR = SeisflowsParameters()
PATH = SeisflowsPaths()

import system
import solver


class ewf2d(custom_import('postprocess', 'base')):
    """ Gradient postprocessing class
    """

    def check(self):
        """ Checks parameters and paths
        """

        # check parameters

        if 'SMOOTH' not in PAR:
            setattr(PAR, 'SMOOTH', 0.)

        if 'USE_PRECOND' not in PAR:
            setattr(PAR, 'USE_PRECOND', False)

        if PAR.USE_PRECOND:
            if 'PRECOND_SMOOTH' not in PAR:
                setattr(PAR, 'USE_PRECOND', 0.0)

            if 'PRECOND_TYPE' not in PAR:
                raise ParameterError(PAR, 'PRECOND_TYPE')

            if PAR.PRECOND_TYPE == 'READ':
                # check path
                if 'PRECOND' not in PATH:
                    raise ParameterError(PATH, 'PRECOND')

                if 'PRECOND_FILE' not in PAR:
                    raise ParameterError(PAR, 'PRECOND_FILE')


    def setup(self):
        """ Performs any required initialization or setup tasks
        """
        src = glob(join(PATH.MODEL_INIT, '*.bin'))
        dst = join(PATH.MODELS, 'model_est')

        if not exists(dst):
            unix.mkdir(dst)

        unix.cp(src, dst)

    def write_gradient(self, path):
        """ Reads kernels and writes gradient of objective function
        """
        if not exists(path):
            unix.mkdir(path)

        self.combine_kernels(path, solver.parameters)
        self.process_kernels(path, solver.parameters)

    def combine_kernels(self, path, parameters):
        system.run('solver', 'combine',
                   hosts='head',
                   precond=PAR.USE_PRECOND)

    def process_kernels(self, path, parameters):
        if PAR.SMOOTH > 0.:
            system.run('solver', 'smooth',
                       hosts='head',
                       span=PAR.SMOOTH)



