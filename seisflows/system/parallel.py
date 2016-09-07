
import os
import subprocess
from os.path import abspath, basename, join

import numpy as np

from seisflows.tools import unix
from seisflows.tools.code import findpath, saveobj

from seisflows.tools.config import SeisflowsParameters, SeisflowsPaths, \
    ParameterError, custom_import
from seisflows.tools.msg import mpiError1, mpiError3


PAR = SeisflowsParameters()
PATH = SeisflowsPaths()


class parallel(custom_import('system', 'mpi_queue')):
    """ An interface through which to submit workflows, run tasks in serial or 
      parallel, and perform other system functions.

      By hiding environment details behind a python interface layer, these 
      classes provide a consistent command set across different computing
      environments.

      For important additional information, please see 
      http://seisflows.readthedocs.org/en/latest/manual/manual.html#system-configuration
    """

    def check(self):
        """ Checks parameters and paths
        """

        if 'TITLE' not in PAR:
            setattr(PAR, 'TITLE', basename(abspath('.')))

        if 'NTASK' not in PAR:
            setattr(PAR, 'NTASK', 1)

        if 'NPROC' not in PAR:
            setattr(PAR, 'NPROC', 1)

        if 'VERBOSE' not in PAR:
            setattr(PAR, 'VERBOSE', 1)

        if 'MPIEXEC' not in PAR:
            setattr(PAR, 'MPIEXEC', 'mpiexec')

        if 'MPIARGS' not in PAR:
            setattr(PAR, 'MPIARGS', '--mca mpi_warn_on_fork 0')

        # check paths
        if 'SCRATCH' not in PATH:
            setattr(PATH, 'SCRATCH', join(abspath('.'), 'scratch'))

        if 'LOCAL' not in PATH:
            setattr(PATH, 'LOCAL', '')

        if 'SUBMIT' not in PATH:
            setattr(PATH, 'SUBMIT', abspath('.'))

        if 'OUTPUT' not in PATH:
            setattr(PATH, 'OUTPUT', join(PATH.SUBMIT, 'output'))

        self.check_mpi()


    def run(self, classname, funcname, hosts='all', **kwargs):
        """ Runs tasks in serial or parallel on specified hosts
        """
        # to avoid cryptic MPI messages, use "--mca_warn_on_fork 0" as the
        # default value for MPIARGS, and use subprocess.call rather than
        # call_catch to invoke mpiexec
        self.checkpoint()
        self.save_kwargs(classname, funcname, kwargs)

        if hosts == 'all':
            unix.cd(join(findpath('seisflows.system'), 'wrappers'))
            subprocess.call(PAR.MPIEXEC + ' '
                    + '-n {} '.format(PAR.NPROCMAX)
                    + PAR.MPIARGS + ' '
                    + 'run_mpi_loop' + ' '
                    + PATH.OUTPUT + ' '
                    + classname + ' '
                    + funcname + ' '
                    + '{}'.format(int(PAR.NTASK / PAR.NPROCMAX)),
                    shell=True)

        elif hosts == 'head':
            unix.cd(join(findpath('seisflows.system'), 'wrappers'))
            subprocess.call(PAR.MPIEXEC + ' '
                    + '-n 1 '
                    + PAR.MPIARGS + ' '
                    + 'run_mpi_head' + ' '
                    + PATH.OUTPUT + ' '
                    + classname + ' '
                    + funcname,
                    shell=True)

        elif hosts == 'mpi_c':
            func = getattr(__import__(classname), funcname)
            func(**kwargs)

        else:
            raise(KeyError('Hosts parameter not set/recognized.'))

    def mpiexec(self):
        """ Specifies MPI exectuable; used to invoke solver
        """
        if PAR.NPROC > 1:
            return 'mpiexec -np %d ' % PAR.NPROC
        else:
            return ''

    def check_mpi(self):
        """ Checks MPI dependencies
        """
        try:
            import mpi4py
        except ImportError:
            raise Exception(mpiError1 % PAR.SYSTEM)

        if PAR.NTASK % PAR.NPROCMAX != 0:
            raise Exception(mpiError3.format(PAR.SYSTEM))
