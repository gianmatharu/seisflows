
import sys
from glob import glob
from os.path import join

from seisflows.tools import unix
from seisflows.tools.tools import exists
from seisflows.plugins.solver_io.pewf2d import mread, mwrite
from seisflows.config import ParameterError, custom_import

PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']

system = sys.modules['seisflows_system']
solver = sys.modules['seisflows_solver']


class pewf2d(custom_import('postprocess', 'base')):
    """ Postprocessing class for pewf2d

        Postprocesing refers to image processing and regularization operations on
        models or gradients
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
        if 'MASK' not in PATH:
            setattr(PATH, 'MASK', None)

        if 'PRECOND' not in PATH:
            setattr(PATH, 'PRECOND', None)

        if PATH.MASK:
            assert exists(PATH.MASK)

    def setup(self):
        """ Performs any required initialization or setup tasks
        """
        src = glob(join(PATH.MODEL_INIT, '*.bin'))
        dst = PATH.MODEL_EST

        if not exists(dst):
            unix.mkdir(dst)

        unix.cp(src, dst)

    def write_gradient(self, path, solver_path=''):
        """ Reads kernels and writes gradient of objective function
        """
        if not exists(path):
            unix.mkdir(path)

        system.run_single('postprocess', 'process_kernels',
                          path=path,
                          parameters=solver.parameters,
                          solver_path=solver_path)

        gradient = solver.load(path, suffix='_kernel')

        if PAR.RESCALE:
            self.rescale_kernels(gradient)

        g = solver.merge(gradient)

        if PAR.RESCALE:
            self.save(path, g, backup='noscale')

        if PATH.MASK:
            # apply mask
            g *= solver.merge(solver.load(PATH.MASK))
            self.save(path, g, backup='nomask')

    def process_kernels(self, path, parameters, solver_path=''):
        """ Process kernels
        """
        solver.combine(path=path,
                       solver_path=solver_path,
                       parameters=parameters)

        if PAR.SMOOTH > 0.:
            solver.smooth(path=path,
                          parameters=parameters,
                          span=PAR.SMOOTH)

    def rescale_kernels(self, gradient):
        """ Applies rescaling to the gradient.
            Correction non-dimensionalized parameters (m' = m / scale)
        """
        for key in solver.parameters:
            gradient[key] *= solver.scale[key]

    def save(self, path, g, backup=None):
        if backup:
            for par in solver.parameters:
                src = path + '/' + '{}_kernel.bin'.format(par)
                dst = path + '/' + '{}_kernel_{}.bin'.format(par, backup)
                unix.mv(src, dst)

        solver.save(solver.split(g), path, suffix='_kernel')


