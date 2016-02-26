from glob import glob
from os.path import join
import sys
import numpy as np

from seisflows.seistools.ewf2d import iter_dirname, event_dirname
from seisflows.tools import msg
from seisflows.tools import unix
from seisflows.tools.array import loadnpy, savenpy
from seisflows.tools.code import divides, exists
from seisflows.tools.config import SeisflowsParameters, SeisflowsPaths, \
    ParameterError, loadclass

PAR = SeisflowsParameters()
PATH = SeisflowsPaths()

import system
import solver
import optimize
import preprocess
import postprocess


class alt_inversion(loadclass('workflow', 'inversion')):
    """ Seismic inversion base class.

      Compute iterative non-linear inversion. Designed to fit EWF2D solver.
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

        super(alt_inversion, self).check()

        # check paths
        if 'MODELS' not in PATH:
            setattr(PATH, 'MODELS', join(PATH.SUBMIT, 'models'))

        if 'MODEL_TRUE' not in PATH:
            raise ParameterError(PATH, 'MODEL_TRUE')

        if 'MODEL_EST' not in PATH:
            setattr(PATH, 'MODEL_EST', join(PATH.MODELS, 'model_est'))

    def main(self):
        """ Carries out seismic inversion
        """
        optimize.iter = PAR.BEGIN
        self.setup()

        while optimize.iter <= PAR.END:
            print "Starting iteration", optimize.iter
            self.compute_gradient()

            print "Computing search direction"
            self.compute_direction()

            print "Computing step length"
            self.line_search()

            self.finalize()
            self.clean()

            optimize.iter += 1
            print ''


    def setup(self):
        """ Lays groundwork for inversion
        """
        # clean scratch directories
        if PAR.BEGIN == 1:
            unix.rm(PATH.GLOBAL)
            unix.mkdir(PATH.GLOBAL)

            preprocess.setup()
            postprocess.setup()
            optimize.setup()

        if PATH.DATA:
            print('Copying data...')
        else:
            print('Generating data...')
        system.run('solver', 'setup', 
                   hosts='all')


    def compute_gradient(self):
        """ Compute gradients. Designed to avoid excessive storage
            of boundary files.
        """

        # output for inversion history
        unix.mkdir(join(PATH.OUTPUT, iter_dirname(optimize.iter)))

        # compute gradients
        system.run('solver', 'compute_gradient',
                   hosts='all')
        postprocess.write_gradient(PATH.GRAD)

        # evaluate misfit function
        self.sum_residuals(path=PATH.SOLVER, suffix='new')


    def compute_direction(self):
        """ Computes search direction
        """
        optimize.compute_direction()


    def line_search(self):
        """ Conducts line search in given search direction
        """
        optimize.initialize_search()

        while True:
            self.iterate_search()

            if optimize.isdone:
                optimize.finalize_search()
                break
            elif optimize.step_count < PAR.STEPMAX:
                optimize.compute_step()
                continue
            else:
                retry = optimize.retry_status()
                if retry:
                    print ' Line search failed...\n\n Retrying...'
                    optimize.restart()
                    self.line_search()
                    break
                else:
                    print ' Line search failed...\n\n Aborting...'
                    sys.exit(-1)


    def iterate_search(self):
        """ First, calls self.evaluate_function, which carries out a forward 
          simulation given the current trial model. Then calls
          optimize.update_status, which maintains search history and checks
          stopping conditions.
        """
        if PAR.VERBOSE > 1:
            print " trial step", optimize.step_count+1

        self.evaluate_function()
        optimize.update_status()


    def evaluate_function(self):
        """ Performs forward simulation to evaluate objective function
        """
        self.write_model(path=PATH.FUNC, suffix='try')

        system.run('solver', 'evaluate_function',
                   hosts='all',
                   path=PATH.FUNC)

        self.sum_residuals(path=PATH.FUNC, suffix='try')


    def finalize(self):
        """ Saves results from current model update iteration
        """
        system.checkpoint()

        # save new model to working dir
        src = join(PATH.OPTIMIZE, 'm_new')
        dst = PATH.MODELS
        solver.split(src, dst, '.bin')

        if divides(optimize.iter, PAR.SAVEMODEL):
            self.save_model()

        if divides(optimize.iter, PAR.SAVEGRADIENT):
            self.save_gradient()

        if divides(optimize.iter, PAR.SAVEKERNELS):
            self.save_kernels()

        if divides(optimize.iter, PAR.SAVETRACES):
            self.save_traces()
            self.save_traces(type='syn')

        if divides(optimize.iter, PAR.SAVERESIDUALS):
            self.save_residuals()


    def clean(self):
        """ Cleans directories in which function and gradient evaluations were
          carried out
        """
        unix.rm(PATH.GRAD)
        unix.rm(PATH.FUNC)
        unix.mkdir(PATH.GRAD)
        unix.mkdir(PATH.FUNC)


    def write_model(self, path='', suffix=''):
        """ Writes model in format used by solver
        """
        unix.mkdir(path)
        src = PATH.OPTIMIZE +'/'+ 'm_' + suffix
        dst = path +'/'+ 'model'
        unix.mkdir(dst)
        unix.cp(glob(join(PATH.MODELS, '*.bin')), dst)
        solver.split(src, dst, '.bin')


    def sum_residuals(self, path='', suffix=''):
        """ Returns sum of squares of residuals
        """
        dst = PATH.OPTIMIZE +'/'+ 'f_' + suffix
        residuals = []

        for itask in range(PAR.NTASK):
            src = path +'/'+ event_dirname(itask + 1) +'/'+ 'residuals'
            fromfile = np.loadtxt(src)
            residuals.append(fromfile**2.)
        np.savetxt(dst, [np.sum(residuals)])


    def save_gradient(self):
        src = glob(join(PATH.GRAD, '*_kernel.bin'))
        dst = join(PATH.OUTPUT, iter_dirname(optimize.iter), 'gradient')
        unix.mkdir(dst)
        unix.mv(src, dst)


    def save_model(self):
        src = join(PATH.OPTIMIZE, 'm_new')
        dst = join(PATH.MODELS, 'm{:02d}'.format(optimize.iter))
        unix.mkdir(dst)
        solver.split(src, dst, '.bin')


    def save_kernels(self):
        raise NotImplementedError


    def save_traces(self, type='obs'):
        for itask in range(PAR.NTASK):
            src = glob(join(PATH.SOLVER, 'traces', type, '*.su'))
            dst = join(PATH.OUTPUT, iter_dirname(optimize.iter), event_dirname(itask + 1), type)
            unix.mkdir(dst)
            unix.mv(src, dst)


    def save_residuals(self):
        raise NotImplementedError
        src = join(PATH.GRAD, 'residuals')
        dst = join(PATH.OUTPUT, 'residuals_%04d' % optimize.iter)
        unix.mv(src, dst)



