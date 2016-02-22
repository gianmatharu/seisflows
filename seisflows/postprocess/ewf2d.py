
from glob import glob
from os.path import join
import numpy as np

from seisflows.tools import unix
from seisflows.tools.array import savenpy
from seisflows.tools.code import exists
from seisflows.tools.config import SeisflowsParameters, SeisflowsPaths, \
    ParameterError, loadclass

PAR = SeisflowsParameters()
PATH = SeisflowsPaths()

import system
import solver


class ewf2d(loadclass('postprocess', 'base')):
    """ Gradient postprocessing class
    """

    def check(self):
        """ Checks parameters and paths
        """
        # check parameters

        if 'SMOOTH' not in PAR:
            setattr(PAR, 'SMOOTH', 0.)

        if 'MASK' not in PAR:
            setattr(PAR, 'MASK', False)

    def setup(self):
        """ Performs any required initialization or setup tasks
        """
        src = glob(join(PATH.MODEL_INIT, '*.bin'))
        dst = join(PATH.MODELS, 'model_est')
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
                       precond=PAR.MASK,
                       span=PAR.SMOOTH)



