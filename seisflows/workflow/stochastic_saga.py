
import sys
from glob import glob
from os.path import join

from seisflows.tools import unix
from seisflows.tools.tools import Struct, exists
from seisflows.tools.err import ParameterError

PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']

system = sys.modules['seisflows_system']
solver = sys.modules['seisflows_solver']
optimize = sys.modules['seisflows_optimize']
preprocess = sys.modules['seisflows_preprocess']
postprocess = sys.modules['seisflows_postprocess']

from seisflows.workflow.inversion import inversion

class stochastic_saga(inversion):
    """ Stochastic optimization (SAGA algorithm) subclass
    """
    def check(self):
        """ Checks parameters, paths, and dependencies
        """
        super(stochastic_saga, self).check()

        if 'GRAD_INIT' not in PATH:
            raise ParameterError('PATH.GRAD_INIT not defined. Requires gradient first iteration gradient')

        if 'GRAD_AGG' not in PATH:
            setattr(PATH, 'GRAD_AGG', join(PATH.WORKDIR, 'GRAD_AGG'))

        if 'NSRC_SUBSET' not in PAR:
            raise ParameterError

        if 'SAMPLING' not in PAR:
            setattr(PAR, 'SAMPLING', 'random')

        assert PAR.NTASK == PAR.NSRC_SUBSET


    def setup(self):
        """ Lays groundwork for inversion
        """
        if PAR.BEGIN == 1:
            # copy initial gradient to working directory
            unix.cp(PATH.GRAD_INIT, PATH.GRAD_AGG)

            preprocess.setup()
            postprocess.setup()
            optimize.setup()

        for isrc in range(PAR.NSRC):
            solver.setup(subset=[isrc])


    def initialize(self):

        # choose subset
        solver.generate_subset()

        self.write_model(path=PATH.GRAD, suffix='new')

        print 'Generating synthetics'
        system.run('solver', 'eval_func',
                   path=PATH.GRAD)

        self.write_misfit(path=PATH.GRAD, suffix='new')


    def finalize(self):
        """ Update stored aggregate gradient.
        """
        super(stochastic_saga, self).finalize()

        # update aggregate gradient
        print 'Updating aggregate gradient...'
        solver.update_aggregate_gradient()


    def write_misfit(self, path='', suffix=''):
        """ Writes misfit in format expected by nonlinear optimization library
        """
        src = glob(path +'/'+ 'residuals/*')
        dst = 'f_'+suffix
        total_misfit = preprocess.sum_residuals(src)
        optimize.savetxt(dst, total_misfit)

