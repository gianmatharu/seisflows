
import sys
from os.path import join

from seisflows.tools import unix
from seisflows.tools.array import savenpy
from seisflows.config import ParameterError, custom_import

PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']

system = sys.modules['seisflows_system']
solver = sys.modules['seisflows_solver']
optimize = sys.modules['seisflows_optimize']
preprocess = sys.modules['seisflows_preprocess']
postprocess = sys.modules['seisflows_postprocess']


class frugal_inversion(custom_import('workflow', 'p_inversion')):
    """ Seismic inversion base class.

        Frugal inversion saves wavefield and data during line search
        function evaluations to prevent re-computation during gradient
        computation. Only useable with a backtracking linesearch. If
        conditions are not met, the class launches a standard inversion.
    """

    def solver_status(self, maxiter=1):
        """ Keeps track of whether a forward simulation would be redundant
        """
        if optimize.iter <= maxiter:
            # forward simulation not redundant because solver files do not exist
            # prior to first iteration
            return False

        elif optimize.iter == PAR.BEGIN:
            # forward simulation not redundant because solver files need to be
            # reinstated after possible multiscale transition
            return False

        elif PAR.LINESEARCH != 'Backtrack':
            # thrifty inversion only implemented for backtracking line search,
            # not bracketing line search
            return False

        elif optimize.restarted:
            # forward simulation not redundant following optimization algorithm
            # restart
            return False

        else:
            # if none of the above conditions are triggered, then forward
            # simulation is redundant, can be skipped
            return True


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

        isready = self.solver_status()
        if not isready:
            # initialize directories
            system.run('solver', 'setup',
                       hosts='all')

            # copy/generate data
            if PATH.DATA:
                print('Copying data...')
            else:
                print('Generating data...')
                system.run('solver', 'generate_data',
                            hosts='head')


    def compute_gradient(self):
        """ Compute gradients. Designed to avoid excessive storage
            of boundary files.
        """
        # are prerequisites for gradient evaluation in place?
        isready = self.solver_status(maxiter=2)

        # if not, then prepare for gradient evaluation
        if not isready:
            super(frugal_inversion, self).compute_gradient()
        else:
            print('Computing gradient (frugal)...')

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


    def evaluate_function(self):
        """ Performs forward simulation to evaluate objective function
        """
        print('Frugal eval...')
        self.write_model(path=PATH.FUNC, suffix='try')

        system.run('solver', 'evaluate_function',
                   mode=1,
                   hosts='head')
        system.run('solver', 'process_trial_step',
                   hosts='all')

        self.sum_residuals(path=PATH.FUNC, suffix='try')


    def iterate_search(self):
        """ First, calls self.evaluate_function, which carries out a forward
          simulation given the current trial model. Then calls
          optimize.update_status, which maintains search history and checks
          stopping conditions.
        """
        super(frugal_inversion, self).iterate_search()

        isdone = optimize.isdone
        isready = self.solver_status()

        # to avoid redundant forward simulation, save solver files associated
        # with 'best' trial model
        if isready and isdone:
            system.run('solver', 'export_trial_solution',
                       hosts='all',
                       path=PATH.FUNC)