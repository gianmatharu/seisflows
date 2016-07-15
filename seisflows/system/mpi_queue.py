
import os
from os.path import abspath, basename, join

import numpy as np
import subprocess

from seisflows.tools import unix
from seisflows.tools.code import findpath, saveobj
from seisflows.tools.config import SeisflowsParameters, SeisflowsPaths, \
    ParameterError, custom_import
from seisflows.tools.msg import mpiError1, mpiError2, mpiError3

PAR = SeisflowsParameters()
PATH = SeisflowsPaths()


class mpi_queue(custom_import('system', 'mpi')):
    """ An interface through which to submit workflows, run tasks in serial or
      parallel, and perform other system functions.

      By hiding environment details behind a python interface layer, these
      classes provide a consistent command set across different computing
      environments.

      For more informations, see
      http://seisflows.readthedocs.org/en/latest/manual/manual.html#system-interfaces
    """

    def check(self):
        """ Checks parameters and paths
        """

        super(mpi_queue, self).check()

        if 'NPROCMAX' not in PAR:
            raise ParameterError(PAR, 'NPROCMAX')

    def submit(self, workflow):
        """ Submits job
        """
        unix.mkdir(PATH.OUTPUT)
        unix.cd(PATH.OUTPUT)

        self.checkpoint()
        workflow.main()

    def run(self, classname, funcname, hosts='all', **kwargs):
        """ Runs tasks in serial or parallel on specified hosts
        """

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
            os.environ['SEISFLOWS_TASKID'] = str(0)
            unix.cd(join(findpath('seisflows.system'), 'wrappers'))
            subprocess.call(PAR.MPIEXEC + ' '
                    + '-n 1 '
                    + PAR.MPIARGS + ' '
                    + 'run_mpi_head' + ' '
                    + PATH.OUTPUT + ' '
                    + classname + ' '
                    + funcname,
                    shell=True)

        else:
            raise(KeyError('Hosts parameter not set/recognized.'))

    def getnode(self):
        """Gets number of running task"""
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
        nproc = int(PAR.NTASK / PAR.NPROCMAX)

        rank_file = join(PATH.SUBMIT, 'rank_{:03d}'.format(rank))

        with open(rank_file) as f:
            ind = int(f.read())
        return (rank * nproc) + ind

    def check_mpi(self):
        """ Checks MPI dependencies
        """
        try:
            import mpi4py
        except ImportError:
            raise Exception(mpiError1 % PAR.SYSTEM)

        if PAR.NPROC > 1:
            raise Exception(mpiError2 % PAR.SYSTEM)

        if PAR.NTASK % PAR.NPROCMAX != 0:
            raise Exception(mpiError3.format(PAR.SYSTEM))

