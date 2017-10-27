
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
    status = 0

    def update_status(self, maxiter=1, optimize_isdone=False):
        """ Keeps track of whether a forward simulation would be redundant
        """
        if optimize.iter <= maxiter:
            # forward simulation not redundant because solver files do not exist
            # prior to first iteration
            self.status = 0

        elif not optimize_isdone:
            if optimize.iter == PAR.BEGIN:
                # forward simulation not redundant because solver files need to be
                # reinstated after possible multiscale transition
                self.status = 0

        elif PAR.LINESEARCH != 'Backtrack':
            # thrifty inversion only implemented for backtracking line search,
            # not bracketing line search
            self.status = 0

        elif optimize.restarted:
            # forward simulation not redundant following optimization algorithm
            # restart
            self.status = 0

        else:
            # if none of the above conditions are triggered, then forward
            # simulation is redundant, can be skipped
            self.status = 1


    def compute_gradient(self):
        """ Compute gradients. Designed to avoid excessive storage
            of boundary files.
        """
        # are prerequisites for gradient evaluation in place?
        self.update_status()

        # if not, then prepare for gradient evaluation
        if self.status == 0:
            super(frugal_inversion, self).compute_gradient()
        else:
            print('Computing gradient (frugal)...')
            print('Prepare adjoint sources...')
            system.run('solver', 'prepare_eval_grad')

            print('Computing gradient...')
            system.run_single('solver', 'compute_gradient')

            postprocess.write_gradient(PATH.GRAD)
            dst = join(PATH.OPTIMIZE, 'g_new')
            savenpy(dst, solver.merge(solver.load(PATH.GRAD, suffix='_kernel')))

            # evaluate misfit function
            self.write_misfit(path=PATH.SOLVER, suffix='new')


    def evaluate_function(self):
        """ Performs forward simulation to evaluate objective function
        """
        print('Frugal eval...')
        self.write_model(path=PATH.FUNC, suffix='try')

        system.run_single('solver', 'evaluate_function', mode=1)
        system.run('solver', 'process_trial_step')

        self.write_misfit(path=PATH.FUNC, suffix='try')


    def line_search(self):
        """ Conducts line search in given search direction

            Status codes
                status > 0  : finished
                status == 0 : not finished
                status < 0  : failed
          """
        optimize.initialize_search()

        while True:
            print " trial step", optimize.line_search.step_count + 1
            self.evaluate_function()
            status = optimize.update_search()

            if status > 0:
                self.update_status(optimize_isdone=status)
                if self.status == 1:
                    system.run('solver', 'export_trial_solution', path=PATH.FUNC)
                optimize.finalize_search()
                break

            elif status == 0:
                continue

            elif status < 0:
                if optimize.retry_status():
                    print ' Line search failed\n\n Retrying...'
                    optimize.restart()
                    self.line_search()
                    break
                else:
                    print ' Line search failed\n\n Aborting...'
                    sys.exit(-1)