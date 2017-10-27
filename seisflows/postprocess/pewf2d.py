
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
        if 'MASK' not in PATH:
            setattr(PATH, 'MASK', None)

        if 'PRECOND' not in PATH:
            setattr(PATH, 'PRECOND', None)

        if PATH.MASK:
            assert exists(PATH.MASK)

    def setup(self):
        """ Performs any required initialization or setup tasks
        """
        if solver.reparam:
            model = mread(PATH.MODEL_INIT, ['vp','vs','rho'])
            model = solver.par_map_forward(model)
            mwrite(model, PATH.MODEL_INIT)

        src = glob(join(PATH.MODEL_INIT, '*.bin'))
        dst = join(PATH.MODELS, 'model_est')

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

        g = solver.merge(solver.load(path, suffix='_kernel'))

        if PATH.MASK:
            # apply mask
            g *= solver.merge(solver.load(PATH.MASK))
            self.save(path, g, backup='nomask')

    def process_kernels(self, path, parameters, solver_path=''):
        solver.combine(path=path,
                       solver_path=solver_path,
                       parameters=parameters)

        if PAR.SMOOTH > 0.:
            solver.smooth(path=path,
                          parameters=parameters,
                          span=PAR.SMOOTH)

    def save(self, path, g, backup=None):
        if backup:
            for par in solver.parameters:
                src = path + '/' + '{}_kernel.bin'.format(par)
                dst = path + '/' + '{}_kernel_{}.bin'.format(par, backup)
                unix.mv(src, dst)

        solver.save(solver.split(g), path, suffix='_kernel')


