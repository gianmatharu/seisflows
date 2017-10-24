import sys
from glob import glob
from os.path import basename, join

import numpy as np

from seisflows.tools import unix
from seisflows.tools.tools import exists
from seisflows.tools.seismic import call_solver

from seisflows.config import ParameterError, custom_import

PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']

system = sys.modules['seisflows_system']


class source_decimation(custom_import('solver', 'base')):

    def setup(self, subset=[]):
        for isrc in subset:
            setattr(self, '_source_subset', [isrc])
            super(source_decimation, self).setup()


    def generate_subset(self):
        setattr(self, '_source_subset',
                np.random.permutation(range(PAR.NSRC))[:PAR.NSRC_SUBSET])
        print self._source_subset

    @property
    def taskid(self):
        try:
            ii = system.taskid()
        except:
            ii = 0
        return ii

    @property
    def source_name(self):
        # returns name of source currently under consideration
        ii = self.taskid
        jj = self._source_subset[ii]
        return self.source_names[jj]

    @property
    def cwd(self):
        assert hasattr(self, '_source_subset')

        ii = self.taskid
        jj = self._source_subset[ii]

        name = self.check_source_names()[jj]
        return join(PATH.SOLVER, name)


    def check_source_names(self):
        """ Checks names of sources
        """
        if not hasattr(self, '_source_names'):
            path = PATH.SPECFEM_DATA
            wildcard = self.source_prefix+'_*'
            globstar = sorted(glob(path +'/'+ wildcard))
            if not globstar:
                 print msg.SourceError_SPECFEM % (path, wildcard)
                 sys.exit(-1)
            names = []
            for path in globstar:
                names += [basename(path).split('_')[-1]]
            self._source_names = names[:PAR.NSRC]

        return self._source_names


    def combine(self, input_path='', output_path='', parameters=[]):
        """ Sums individual source contributions. Wrapper over xcombine_sem
            utility.
        """
        if not exists(input_path):
            raise Exception

        if not exists(output_path):
            unix.mkdir(output_path)

        unix.cd(self.cwd)

        names = self.check_source_names()
        subset = [names[isrc] for isrc in self._source_subset]

        with open('kernel_paths', 'w') as f:
            f.writelines([join(input_path, dir)+'\n' for dir in subset])

        for name in parameters or self.parameters:
            call_solver(
                system.mpiexec(),
                PATH.SPECFEM_BIN +'/'+ 'xcombine_sem '
                + name + '_kernel' + ' '
                + 'kernel_paths' + ' '
                + output_path)