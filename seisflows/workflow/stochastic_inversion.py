
import sys
from os.path import join
from glob import glob

from seisflows.tools import unix
from seisflows.tools.array import loadnpy, savenpy
from seisflows.tools.tools import saveobj, divides, exists
from seisflows.config import ParameterError, custom_import
from seisflows.plugins.solver.pewf2d import iter_dirname, event_dirname

PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']

system = sys.modules['seisflows_system']
solver = sys.modules['seisflows_solver']
optimize = sys.modules['seisflows_optimize']
preprocess = sys.modules['seisflows_preprocess']
postprocess = sys.modules['seisflows_postprocess']


class stochastic_inversion(custom_import('workflow', 'p_inversion')):
    """ Seismic inversion base class.

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
        super(stochastic_inversion, self).check()

        if not PATH.DATA:
            raise ValueError('Data path required.')

        if PAR.PREPROCESS != 'pewf2d':
            raise ValueError('Can only run stochastic inversion with pewf2d preprocessing class.')

        if PAR.SOLVER != 'spewf2d':
            raise ValueError('Can only run stochastic inversion with spewf2d solver class.')

        if 'SAVESUBSET' not in PAR:
            setattr(PAR, 'SAVESUBSET', 1)

        if PAR.OPTIMIZE == 'NLCG':
            if PAR.NLCGMAX != PAR.ITER_RESET:
                raise ValueError('Encoding reset must match optimization reset.')
        elif PAR.OPTIMIZE == 'LBFGS':
            if PAR.LBFGSMAX != PAR.ITER_RESET:
                raise ValueError('Encoding reset must match optimization reset.')

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

        system.run('solver', 'setup_sources',
                   hosts='head')

    def compute_gradient(self):
        """ Compute gradients. Designed to avoid excessive storage
            of boundary files.
        """
        # output for inversion history
        unix.mkdir(join(PATH.OUTPUT, iter_dirname(optimize.iter)))

        if optimize.iter == 1 or (optimize.iter % PAR.ITER_RESET == 0):
            print('Set decimated sources...')
            system.run('solver', 'select_sources',
                       hosts='head')

            print('Fetching data...')
            system.run('solver', 'fetch_data',
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

        dst = join(PATH.OPTIMIZE, 'g_new')
        savenpy(dst, solver.merge(solver.load(PATH.GRAD,
                                              suffix='_kernel')))

        # evaluate misfit function
        self.sum_residuals(path=PATH.SOLVER, suffix='new')

    def finalize(self):
        """ Saves results from current model update iteration
        """
        system.checkpoint()

        if divides(optimize.iter, PAR.SAVESUBSET):
            self.save_subset()

        super(stochastic_inversion, self).finalize()

    def save_subset(self):
        """ Save encoding scheme.
        """
        # copy source files
        src = glob(join(PATH.SOURCE, 'SOURCES'))
        dst = join(PATH.OUTPUT, iter_dirname(optimize.iter))
        unix.cp(src, dst)

