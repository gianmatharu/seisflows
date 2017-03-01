
import sys
from glob import glob
from os.path import join

from seisflows.tools import unix
from seisflows.tools.tools import exists
from seisflows.config import ParameterError, custom_import

PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']

system = sys.modules['seisflows_system']
solver = sys.modules['seisflows_solver']


class pewf2d(custom_import('postprocess', 'base')):
    """ Gradient postprocessing class
    """

    def check(self):
        """ Checks parameters and paths
        """

        # check parameters
        if 'SMOOTH' not in PAR:
            setattr(PAR, 'SMOOTH', 0.)

        if 'PRECOND' not in PAR:
            setattr(PAR, 'PRECOND', False)

        # check paths
        if 'PRECOND' not in PATH:
            setattr(PATH, 'PRECOND', None)

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
                   hosts='head')

    def process_kernels(self, path, parameters):
        if PAR.SMOOTH > 0.:
            system.run('solver', 'smooth',
                       hosts='head',
                       span=PAR.SMOOTH)



