
import os
from os.path import abspath, basename, join

import numpy as np

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

            iter = 0
            queue = range(PAR.NTASK)

            while True:
                unix.cd(join(findpath('seisflows.system'), 'wrappers'))
                os.environ['SEISFLOWS_TASKID'] = str(iter)
                unix.run('mpiexec -n {} '.format(PAR.NPROCMAX)
                        + PAR.MPIARGS + ' '
                        + 'run_mpi' + ' '
                        + PATH.OUTPUT + ' '
                        + classname + ' '
                        + funcname)

                del queue[:PAR.NPROCMAX]
                if queue:
                    iter += 1
                else:
                    break

        elif hosts == 'head':
            unix.cd(join(findpath('seisflows.system'), 'wrappers'))
            unix.run('mpiexec -n 1 '
                    + PAR.MPIARGS + ' '
                    + 'run_mpi_head' + ' '
                    + PATH.OUTPUT + ' '
                    + classname + ' '
                    + funcname)

        else:
            raise(KeyError('Hosts parameter not set/recognized.'))

    def getnode(self):
        """Gets number of running task"""
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
        iter = int(os.environ['SEISFLOWS_TASKID'])
        return int(iter * PAR.NPROCMAX + rank)

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
            raise Exception(mpiError3 % PAR.SYSTEM)

