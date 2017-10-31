
import sys
from os.path import join, basename
from glob import glob

import numpy as np

from seisflows.tools import unix
from seisflows.plugins.solver_io.pewf2d import read, write
from seisflows.plugins.encode import SourceArray, decimate_source_array
from seisflows.config import ParameterError, custom_import
from seisflows.plugins.solver.pewf2d import  Par, read_cfg_file, write_cfg_file, event_dirname

PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']

system = sys.modules['seisflows_system']
preprocess = sys.modules['seisflows_preprocess']

# Solver parameter class
p = Par()
p.read_par_file(join(PATH.SOLVER_INPUT, 'par_template.cfg'))


class stochastic_newton(custom_import('solver', 'pewf2d')):
    """ Python interface for subsampled truncated Newton method

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

        if 'NSOURCES' not in PAR:
            raise ParameterError(PAR, 'NSOURCES')

        if 'NSUBSET' not in PAR:
            raise ParameterError(PAR, 'NSUBSET')

        if 'STOCHASTIC' not in PAR:
            setattr(PAR, 'STOCHASTIC', True)

        if 'BATCH' not in PAR:
            setattr(PAR, 'BATCH', False)

        if 'VERBOSE' not in PAR:
            setattr(PAR, 'VERBOSE', False)

        # check paths
        if 'SOURCE' not in PATH:
            setattr(PATH, 'SOURCE', join(PATH.WORKDIR, 'sources'))

        # check parameters
        if PAR.NSUBSET > PAR.NSOURCES:
            raise ValueError('Subset must be smaller than set.')

        if PAR.NSUBSET % PAR.NPROC != 0:
            raise NotImplementedError('Subset size must be a multiple of nproc.')

        if PAR.PREPROCESS != 'finite_sum':
            raise ValueError('Use preprocessing class "finite_sum"')


    def apply_hess(self, adjoint=False):
        """ Used to compute action of the Hessian on a model perturbation.
        """
        output_dir = PATH.HESS
        model_dir = join(PATH.HESS, 'model')

        if not adjoint:
            mode=1
            run_solver = self.forward
        else:
            mode = 2
            run_solver = self.adjoint

        self.set_par_cfg(external_model_dir=model_dir,
                         output_dir=output_dir,
                         mode=mode,
                         use_stf_file=PAR.USE_STF_FILE,
                         stf_file='stf_f.txt',
                         use_src_file=True,
                         src_file=join(PATH.SOURCE, 'SOURCES'))

        run_solver()


    def prepare_apply_hess(self):
        """ Prepare adjoint sources
        """
        self.fetch_data()
        for itask in xrange(PAR.NSUBSET):
            path = join(PATH.HESS, event_dirname(itask+1))

            for filename in self.data_filenames:
                obs = preprocess.reader(path +'/'+'traces/obs', filename)
                syn = preprocess.reader(path +'/'+'traces/syn', filename)

                obs = preprocess.process_traces(obs, filter=not PAR.PREFILTER)
                syn = preprocess.process_traces(syn, filter=False)

                preprocess.write_adjoint_traces(path+'/'+'traces/adj', syn, obs, filename)


    # source sampling functions

    def setup_sources(self):
        """ Select samples for stochastic optimization
        """
        # construct source array (all sources)
        self.source_array = SourceArray.fromfile(join(PATH.DATA, PAR.SOURCE_FILE))

        # check input
        if len(self.source_array) != PAR.NSOURCES:
            raise ValueError('NSOURCES in parameter file does not match length of input source file.')


    def select_sources(self):
        """ Sample subset from source array.
        """
        self.setup_sources()
        # generate working source array
        self.source_array_subset = decimate_source_array(self.source_array,
                                                         PAR.NSUBSET,
                                                         random=PAR.STOCHASTIC,
                                                         batch=PAR.BATCH)
        self._write_source_file()

        if PAR.VERBOSE:
            self.source_array.print_positions()
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
                f.write('{:.6f} {:.6f}\n'.format(self.source_array_subset[i].x, self.source_array_subset[i].z))

        return

    def fetch_data(self):
        """ Copy data to working directory.
        """
        for itask in xrange(PAR.NSUBSET):
            ishot = self.source_array_subset[itask].index
            src = glob(PATH.DATA + '/' + event_dirname(ishot) + '/' + '*')
            dst = join(PATH.HESS, event_dirname(itask + 1), 'traces/obs/')
            unix.cp(src, dst)