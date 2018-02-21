
import sys
from os.path import join, basename
from glob import glob

import numpy as np

from seisflows.tools import unix, sampling
from seisflows.tools.array import loadnpy
from seisflows.plugins.solver_io.pewf2d import read, write
from seisflows.plugins.stochastic import SourceArray, subsample
from seisflows.tools.seismic import call_solver
from seisflows.config import ParameterError, custom_import
from seisflows.plugins.solver.pewf2d import Par, event_dirname

PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']

system = sys.modules['seisflows_system']
preprocess = sys.modules['seisflows_preprocess']

# Solver parameter class
p = Par()
p.read_par_file(join(PATH.SOLVER_INPUT, 'par_template.cfg'))


class stochastic_newton(custom_import('solver', 'pewf2d')):
    """ Python interface for subsampled truncated Newton method (PEWF2D)

      See base class for method descriptions.
      PEWF2D class differs in that the solver incorporates shot
      parallelism into the source code.
    """

    def check(self):
        """ Checks parameters and paths
        """
        super(stochastic_newton, self).check()

        # check parameters
        if 'SOURCE_FILE' not in PAR:
            raise ParameterError(PAR, 'SOURCE_FILE')

        # no. sources for gradient evaluation (currently uses full data)
        if 'NSOURCES' not in PAR:
            raise ParameterError(PAR, 'NSOURCES')

        # no. sources for inner CG iteration
        if 'NSUBSET' not in PAR:
            raise ParameterError(PAR, 'NSUBSET')

        # subsampling scheme
        if 'SUBSAMPLING' not in PAR:
            raise ParameterError(PAR, 'SUBSAMPLING')

        if 'VERBOSE' not in PAR:
            setattr(PAR, 'VERBOSE', False)

        # check paths
        if 'SOURCE' not in PATH:
            setattr(PATH, 'SOURCE', join(PATH.WORKDIR, 'sources'))

        # check parameters
        if PAR.NSUBSET > PAR.NSOURCES:
            raise ValueError('Subset must be smaller than set.')

        if PAR.PREPROCESS != 'finite_sum':
            raise ValueError('Use preprocessing class "finite_sum"')

        if PAR.SUBSAMPLING not in dir(sampling):
            raise AttributeError('Sampling scheme not implemented')

        if PAR.SUBSAMPLING == 'non_uniform':
            if 'PROB_DIST_FILE' not in PAR:
                setattr(PAR, 'PROB_DIST_FILE', 'PROB_DIST')

        # set up sources
        self.setup_sources()

        # set up sampling probabilities
        self.setup_prob_dist()

    # low level interface
    def forward_hess(self):
        """ Perform forward simulation. Must launch from /bin.
        """
        unix.cd(PATH.SOLVER_BIN)
        script = './xewf2d'
        call_solver('mpiexec -np {}'.format(PAR.NSUBSET),
                    script,
                    PATH.WORKDIR + '/dump_fwd')

        unix.cd(PATH.WORKDIR)

    def adjoint_hess(self):
        """ Perform adjoint simulation. Must launch from /bin
        """
        unix.cd(PATH.SOLVER_BIN)
        script = './xewf2d'
        call_solver('mpiexec -np {}'.format(PAR.NSUBSET),
                    script,
                    PATH.WORKDIR + '/dump_adj')

        unix.cd(PATH.WORKDIR)

    # higher level interface
    def apply_hess(self, model_dir='', adjoint=False):
        """ Used to compute action of the Hessian on a model perturbation.
        """
        if not adjoint:
            mode=1
            run_solver = self.forward_hess
        else:
            mode = 2
            run_solver = self.adjoint_hess

        self.set_par_cfg(external_model_dir=model_dir,
                         output_dir=PATH.HESS,
                         mode=mode,
                         use_stf_file=PAR.USE_STF_FILE,
                         stf_file='stf_f.txt',
                         use_src_file=True,
                         src_file=join(PATH.SOURCE, 'SOURCES'))

        run_solver()

    def prepare_apply_hess(self):
        """ Prepare adjoint sources
        """
        # moves data to working Hessian directory.
        self.fetch_data()

        # filter data or not
        if PAR.OPTIMIZE == 'stochastic_newton':
            filter_obs = not PAR.PREFILTER
        elif PAR.OPTIMIZE == 'stochastic_gauss_newton':
            filter_obs = False

        for itask in xrange(PAR.NSUBSET):
            path = join(PATH.HESS, event_dirname(itask+1))

            for filename in self.data_filenames:
                obs = preprocess.reader(path +'/'+'traces/obs', filename)
                syn = preprocess.reader(path +'/'+'traces/syn', filename)

                obs = preprocess.process_traces(obs, filter=filter_obs)
                syn = preprocess.process_traces(syn)

                preprocess.write_adjoint_traces(path+'/'+'traces/adj', syn, obs, filename)

    # source sampling functions

    def setup_sources(self):
        """ Select samples for stochastic optimization
        """
        # construct source array (all sources)
        self.source_array = SourceArray.fromfile(join(PATH.DATA, PAR.SOURCE_FILE))

        # check input
        if len(self.source_array) != PAR.NSOURCES:
            raise ValueError('NSOURCES in parameter file does not match length of '
                             'input source file.')

        if PAR.VERBOSE:
            self.source_array.print_positions()

    def setup_prob_dist(self):
        """ Read in non uniform probability distribution
        """
        if PAR.SUBSAMPLING == 'non_uniform':
            self.p_dist = loadnpy(join(PATH.DATA, PAR.PROB_DIST_FILE))
            if np.any(self.p_dist == 0):
                print 'Warning: 0 probability detected for some sources.\n'

            if PAR.VERBOSE:
                print 'Non-uniform sampling probabilities'
                for i in xrange(PAR.NSOURCES):
                    print '{:03d}: {:.4f}'.format(self.source_array[i].index, self.p_dist[i])
                print '\n'
        else:
            self.p_dist = None

    def select_sources(self):
        """ Sample subset from source array.
        """
        # generate working source array
        self.source_array_subset = subsample(self.source_array, PAR.NSUBSET,
                                             scheme=PAR.SUBSAMPLING,
                                             p=self.p_dist)
        self._write_source_file()

        if PAR.VERBOSE:
            print 'Current subset...'
            self.source_array_subset.print_positions()


   # serial/reduction function

    def combine(self, path='', solver_path='', parameters=[]):
        """ sum event gradients to compute misfit gradient
        """
        grad = {}

        if not solver_path:
            solver_path = PATH.SOLVER

        # Change summation depending on call
        if solver_path == PATH.HESS:
            ntask = PAR.NSUBSET
        else:
            ntask = PAR.NSOURCES
        print 'Dividing by: {}'.format(ntask)

        # sum gradient
        for key in parameters or self.parameters:
            gradp = np.zeros(p.nx * p.nz, dtype='float32')
            for itask in range(ntask):
                fpath = join(solver_path, event_dirname(itask + 1), 'traces/syn')
                gradp += read(fpath, key, suffix='_kernel')

            grad[key] = (1.0/ntask) * gradp

            if PAR.RESCALE:
                grad[key] *= self.scale[key]

        # backup raw kernel
        self.save(grad, path, suffix='_kernel')

    def combine_subset(self, path='', parameters=[]):
        """ sum event gradients over a subset of sources
        """
        grad = {}

        # sum gradient
        for key in parameters or self.parameters:
            gradp = np.zeros(p.nx * p.nz, dtype='float32')
            for itask in range(PAR.NSUBSET):
                ishot = self.source_array_subset[itask].index
                fpath = join(PATH.SOLVER, event_dirname(ishot), 'traces/syn')
                gradp += read(fpath, key, suffix='_kernel')

            grad[key] = (1.0/PAR.NSUBSET) * gradp

            if PAR.RESCALE:
                grad[key] *= self.scale[key]

        # backup raw kernel
        self.save(grad, path, suffix='_kernel')

    # utility functions
    def _write_source_file(self):
        """ Write solver suitable source file.
        """
        filename = join(PATH.SOURCE, 'SOURCES')
        with open(filename, 'w') as f:
            for i in xrange(len(self.source_array_subset)):
                f.write('{:.6f} {:.6f}\n'.format(self.source_array_subset[i].x,
                                                 self.source_array_subset[i].z))

        return

    def fetch_data(self):
        """ Copy data to working directory.
            Copies true data for TN and synthetic data for TGN.
        """
        for itask in xrange(PAR.NSUBSET):
            ishot = self.source_array_subset[itask].index
            path = join(PATH.SOLVER, event_dirname(ishot))

            if PAR.OPTIMIZE == 'stochastic_newton':
                src = glob(path+'/traces/obs/*.su')
            elif PAR.OPTIMIZE == 'stochastic_gauss_newton':
                src = [glob(path+'/traces/syn/*.su'),
                       glob(path+'/traces/syn/proc*')]
            else:
                raise ValueError('PAR.OPTIMIZE not recognized.')

            dst = join(PATH.HESS, event_dirname(itask + 1), 'traces/obs/')
            unix.cp(src, dst)