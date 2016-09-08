from os.path import abspath, basename, join, dirname
import os

from seisflows.tools import unix
from seisflows.tools.code import call, findpath, saveobj
from seisflows.tools.config import SeisflowsParameters, SeisflowsPaths, \
    ParameterError, custom_import

PAR = SeisflowsParameters()
PATH = SeisflowsPaths()


class westgrid(custom_import('system', 'base')):
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
        if 'SCRATCH' not in PATH:
            setattr(PATH, 'SCRATCH', join(abspath('.'), 'scratch'))

        if 'LOCAL' not in PATH:
            setattr(PATH, 'LOCAL', None)

        if 'SYSTEM' not in PATH:
            setattr(PATH, 'SYSTEM', join(PATH.SCRATCH, 'system'))

        if 'SUBMIT' not in PATH:
            setattr(PATH, 'SUBMIT', abspath('.'))

        if 'OUTPUT' not in PATH:
            setattr(PATH, 'OUTPUT', join(PATH.SUBMIT, 'output'))


    def submit(self, workflow):
        """Submits job
        """
        unix.mkdir(PATH.OUTPUT)
        unix.cd(PATH.OUTPUT)

        # save current state
        self.checkpoint()
        workflow.main()


    def run(self, classname, funcname, hosts='all', **kwargs):
        """  Runs tasks in serial or parallel on specified hosts
        """
        self.checkpoint()
        self.save_kwargs(classname, funcname, kwargs)

        if hosts == 'all':
            # run on all available nodes

            iter = 0
            queue = list(range(PAR.NTASK))

            while queue:

                self.setnode(iter)

                call('pbsdsh '
                        + join(findpath('seisflows.system'), 'wrappers/export_paths.sh ')
                        + os.getenv('PATH') + ' '
                        + os.getenv('LD_LIBRARY_PATH') + ' '
                        + join(findpath('seisflows.system'), 'wrappers/run_pbsdsh ')
                        + PATH.OUTPUT + ' '
                        + classname + ' '
                        + funcname + ' '
                        + dirname(findpath('seisflows').rstrip('/')))

                iter += 1
                queue = queue[:-PAR.NPROC]

        elif hosts == 'head':
            # run on head node
            call('pbsdsh '
                    + join(findpath('seisflows.system'), 'wrappers/export_paths.sh ')
                    + os.getenv('PATH') + ' '
                    + os.getenv('LD_LIBRARY_PATH') + ' '
                    + join(findpath('seisflows.system'), 'wrappers/run_pbsdsh_head ')
                    + PATH.OUTPUT + ' '
                    + classname + ' '
                    + funcname + ' '
                    + dirname(findpath('seisflows').rstrip('/')))

        elif hosts == 'mpi_c':
            func = getattr(__import__(classname), funcname)
            func(**kwargs)

        else:
            raise(KeyError('Hosts parameter not set/recognized.'))


    def setnode(self, iter):
        """ Sets number of running task
        """
        itask = int(os.getenv('PBS_VNODENUM')) + int(iter * PAR.NPROC)
        os.environ['SEISFLOWS_TASKID'] = str(itask)


    def getnode(self):
        """ Gets number of running task
        """
        #return int(os.getenv('PBS_VNODENUM'))
        return int(os.getenv['SEISFLOWS_TASKID'])


    def save_kwargs(self, classname, funcname, kwargs):
        kwargspath = join(PATH.OUTPUT, 'SeisflowsObjects', classname+'_kwargs')
        kwargsfile = join(kwargspath, funcname+'.p')
        unix.mkdir(kwargspath)
        saveobj(kwargsfile, kwargs)


    def mpiexec(self):
        """ Specifies MPI exectuable; used to invoke solver
        """
        if PAR.NPROC > 1:
            return 'mpiexec -np %d ' % PAR.NPROC
        else:
            return ''