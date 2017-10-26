
import sys
import os
from os.path import abspath, basename, join, dirname

from seisflows.tools import unix
from seisflows.tools.tools import call, findpath, saveobj
from seisflows.config import ParameterError, custom_import
from seisflows.workflow.base import base


PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']


class westgrid(base):
    """ An interface through which to submit workflows, run tasks in serial or
      parallel, and perform other system functions.

      By hiding environment details behind a python interface layer, these
      classes provide a consistent command set across different computing
      environments.

      Intermediate files are written to a global scratch path PATH.SCRATCH,
      which must be accessible to all compute nodes.

      Optionally, users can provide a local scratch path PATH.LOCAL if each
      compute node has its own local filesystem.

      For important additional information, please see
      http://seisflows.readthedocs.org/en/latest/manual/manual.html#system-configuration
    """

    def check(self):
        """ Checks parameters and paths
        """

        # check parameters
        if 'TITLE' not in PAR:
            setattr(PAR, 'TITLE', basename(abspath('.')))

        if 'VERBOSE' not in PAR:
            setattr(PAR, 'VERBOSE', 1)

        if 'NTASK' not in PAR:
            raise ParameterError(PAR, 'NTASK')

        if 'NPROC' not in PAR:
            raise ParameterError(PAR, 'NPROC')

        if PAR.NTASK % PAR.NPROC != 0:
            raise NotImplementedError('NTASK must be a multiple of NPROC')

        if 'PBSARGS' not in PAR:
            setattr(PAR, 'PBSARGS', '')

        # check paths
        if 'WORKDIR' not in PATH:
            setattr(PATH, 'WORKDIR', abspath('.'))

        if 'SCRATCH' not in PATH:
            setattr(PATH, 'SCRATCH', join(abspath('.'), 'scratch'))

        if 'LOCAL' not in PATH:
            setattr(PATH, 'LOCAL', None)

        if 'SYSTEM' not in PATH:
            setattr(PATH, 'SYSTEM', join(PATH.SCRATCH, 'system'))

        if 'OUTPUT' not in PATH:
            setattr(PATH, 'OUTPUT', join(PATH.WORKDIR, 'output'))


    def submit(self, workflow):
        """Submits job
        """
        unix.mkdir(PATH.OUTPUT)
        unix.cd(PATH.OUTPUT)

        # save current state
        self.checkpoint()
        workflow.main()


    def run(self, classname, funcname, **kwargs):
        """  Runs tasks in serial or parallel on all hosts
        """
        self.checkpoint()
        self.save_kwargs(classname, funcname, kwargs)

        # run on all available nodes
        iloop = 0
        queue = list(range(PAR.NTASK))

        while queue:

            call('pbsdsh '
                    + join(findpath('seisflows.system'), 'wrappers/export_paths.sh ')
                    + os.getenv('PATH') + ' '
                    + os.getenv('LD_LIBRARY_PATH') + ' '
                    + str(iloop) + ' '
                    + join(findpath('seisflows.system'), 'wrappers/run_pbsdsh ')
                    + PATH.OUTPUT + ' '
                    + classname + ' '
                    + funcname + ' '
                    + dirname(findpath('seisflows').rstrip('/')))

            iloop += 1
            queue = queue[:-PAR.NPROC]


    def run_single(self, classname, funcname, **kwargs):
        """  Runs tasks in serial on single host
        """
        self.checkpoint()
        self.save_kwargs(classname, funcname, kwargs)
        # run on head node
        call('pbsdsh '
                + join(findpath('seisflows.system'), 'wrappers/export_paths.sh ')
                + os.getenv('PATH') + ' '
                + os.getenv('LD_LIBRARY_PATH') + ' '
                + str(0) + ' '
                + join(findpath('seisflows.system'), 'wrappers/run_pbsdsh_head ')
                + PATH.OUTPUT + ' '
                + classname + ' '
                + funcname + ' '
                + dirname(findpath('seisflows').rstrip('/')))


    def run_parallel(self, classname, funcname, **kwargs):
        """ Run a code that runs in parallel internally.
        """
        func = getattr(__import__('seisflows_'+classname), funcname)
        func(**kwargs)


    def taskid(self):
        """ Gets number of running task
        """
        return int(os.getenv('PBS_VNODENUM')) + int(os.getenv('TASKID')) * PAR.NPROC


    def save_kwargs(self, classname, funcname, kwargs):
        kwargspath = join(PATH.OUTPUT, 'kwargs')
        kwargsfile = join(kwargspath, classname+'_'+funcname+'.p')
        unix.mkdir(kwargspath)
        saveobj(kwargsfile, kwargs)


    def mpiexec(self):
        """ Specifies MPI exectuable; used to invoke solver
        """
        if PAR.NPROC > 1:
            return 'mpiexec -np %d ' % PAR.NPROC
        else:
            return ''