
import os
from os.path import abspath, join

import numpy as np

from seisflows.tools import unix
from seisflows.tools.code import saveobj
from seisflows.tools.config import SeisflowsParameters, SeisflowsPaths, \
    ParameterError, loadclass, findpath

PAR = SeisflowsParameters()
PATH = SeisflowsPaths()


class mpi(loadclass('system', 'base')):
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

        if 'TITLE' not in PAR:
            setattr(PAR, 'TITLE', unix.basename(abspath('..')))

        if 'NTASK' not in PAR:
            setattr(PAR, 'NTASK', 1)

        if 'NPROC' not in PAR:
            setattr(PAR, 'NPROC', 1)

        if 'VERBOSE' not in PAR:
            setattr(PAR, 'VERBOSE', 1)

        # check paths
        if 'SCRATCH' not in PATH:
            setattr(PATH, 'SCRATCH', join(abspath('.'), 'scratch'))

        if 'LOCAL' not in PATH:
            setattr(PATH, 'LOCAL', '')

        if 'SUBMIT' not in PATH:
            setattr(PATH, 'SUBMIT', unix.pwd())

        if 'OUTPUT' not in PATH:
            setattr(PATH, 'OUTPUT', join(PATH.SUBMIT, 'output'))

        if 'SYSTEM' not in PATH:
            setattr(PATH, 'SYSTEM', join(PATH.SCRATCH, 'system'))


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
            unix.cd(join(findpath('system'), 'wrappers'))
            unix.run('mpiexec -n {} '.format(PAR.NTASK)
                    + '--mca mpi_warn_on_fork 0' + ' '
                    + 'run_mpi' + ' '
                    + PATH.OUTPUT + ' '
                    + classname + ' '
                    + funcname)

        elif hosts == 'head':
            unix.cd(join(findpath('system'), 'wrappers'))
            unix.run('mpiexec -n 1 '
                    + '--mca mpi_warn_on_fork 0' + ' '
                    + 'run_mpi_head' + ' '
                    + PATH.OUTPUT + ' '
                    + classname + ' '
                    + funcname)

        else:
            raise(KeyError('Hosts parameter not set/recognized'))

    def getnode(self):
        """Gets number of running task"""
        return int(os.environ['OMPI_COMM_WORLD_RANK'])

    def mpiargs(self):
        """ Wrapper for mpiexec
        """
        return ''

    def save_kwargs(self, classname, funcname, kwargs):
        kwargspath = join(PATH.OUTPUT, 'SeisflowsObjects', classname+'_kwargs')
        kwargsfile = join(kwargspath, funcname+'.p')
        unix.mkdir(kwargspath)
        saveobj(kwargsfile, kwargs)

