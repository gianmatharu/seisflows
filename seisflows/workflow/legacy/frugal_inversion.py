from glob import glob
from os.path import join
import sys
import numpy as np

from seisflows.seistools.ewf2d import iter_dirname, event_dirname
from seisflows.tools import unix
from seisflows.tools.code import divides
from seisflows.tools.config import SeisflowsParameters, SeisflowsPaths, \
    ParameterError, custom_import

PAR = SeisflowsParameters()
PATH = SeisflowsPaths()

import system
import solver
import optimize
import preprocess
import postprocess


class frugal_inversion(custom_import('workflow', 'alt_inversion')):
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
            if PATH.DATA:
                print 'Copying data...'
            else:
                print 'Generating data...'

            system.run('solver', 'setup',
                       hosts='all')

    def compute_gradient(self):
        """ Compute gradients. Designed to avoid excessive storage
            of boundary files.
        """
        # are prerequisites for gradient evaluation in place?
        isready = self.solver_status(maxiter=2)

        # if not, then prepare for gradient evaluation
        if not isready:
            print('Computing gradient...')
            super(frugal_inversion, self).compute_gradient()
        else:
            print('Computing gradient (frugal)...')
            # output for inversion history
            unix.mkdir(join(PATH.OUTPUT, iter_dirname(optimize.iter)))

            # compute gradients
            system.run('solver', 'fg_compute_gradient',
                       hosts='all')
            postprocess.write_gradient(PATH.GRAD)

            # evaluate misfit function
            self.sum_residuals(path=PATH.SOLVER, suffix='new')


    def iterate_search(self):
        super(frugal_inversion, self).iterate_search()

        isdone = optimize.isdone
        isready = self.solver_status()

        # to avoid redundant forward simulation, save solver files associated
        # with 'best' trial model
        if isready and isdone:
            system.run('solver', 'export_trial_solution',
                       hosts='all',
                       path=PATH.FUNC)
