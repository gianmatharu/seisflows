import os
import subprocess
from os.path import abspath, join, dirname

from seisflows.tools import unix
from seisflows.tools.code import saveobj
from seisflows.tools.config import ParameterError, findpath, loadclass, \
    SeisflowsObjects, SeisflowsParameters, SeisflowsPaths

PAR = SeisflowsParameters()
PATH = SeisflowsPaths()


class pbs_wg(loadclass('system', 'base')):
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

        if 'SUBTITLE' not in PAR:
            setattr(PAR, 'SUBTITLE', unix.basename(abspath('.')))

        # check parameters
        if 'WALLTIME' not in PAR:
            setattr(PAR, 'WALLTIME', 30.)

        if 'MEMORY' not in PAR:
            raise ParameterError(PAR, 'MEMORY')

        if 'VERBOSE' not in PAR:
            setattr(PAR, 'VERBOSE', 1)

        if 'NTASK' not in PAR:
            raise ParameterError(PAR, 'NTASK')

        if 'NPROCMAX' not in PAR:
            raise ParameterError(PAR, 'NPROCMAX')

        if 'NPROC' not in PAR:
            raise ParameterError(PAR, 'NPROC')

        if 'NODESIZE' not in PAR:
            raise ParameterError(PAR, 'NODESIZE')

        # check paths
        if 'GLOBAL' not in PATH:
            setattr(PATH, 'GLOBAL', join(abspath('.'), 'scratch'))

        if 'LOCAL' not in PATH:
            setattr(PATH, 'LOCAL', None)

        if 'SYSTEM' not in PATH:
            setattr(PATH, 'SYSTEM', join(PATH.GLOBAL, 'system'))

        if 'SUBMIT' not in PATH:
            setattr(PATH, 'SUBMIT', unix.pwd())

        if 'OUTPUT' not in PATH:
            setattr(PATH, 'OUTPUT', join(PATH.SUBMIT, 'output'))

        if PAR.NTASK % PAR.NPROCMAX != 0:
            raise ValueError('Set procs as multiple of source number, ineffcient')

    def submit(self, workflow):
        """Submits job
        """
        unix.mkdir(PATH.OUTPUT)
        unix.cd(PATH.OUTPUT)

        # save current state
        self.checkpoint()

        # construct resource list
        nodes = int(PAR.NPROCMAX / PAR.NODESIZE)
        cores = PAR.NPROCMAX % PAR.NODESIZE
        hours = int(PAR.WALLTIME / 60)
        minutes = PAR.WALLTIME % 60
        resources = 'walltime=%02d:%02d:00'%(hours, minutes)
        if nodes == 0:
            resources += ',mem=%dgb,nodes=1:ppn=%d'%(PAR.MEMORY, cores)
        elif cores == 0:
            resources += ',mem=%dgb,nodes=%d:ppn=%d'%(PAR.MEMORY, nodes, PAR.NODESIZE)
        else:
            resources += ',mem=%dgb,nodes=%d:ppn=%d+1:ppn=%d'%(PAR.MEMORY, nodes, PAR.NODESIZE, cores)

        # construct arguments list
        unix.run('qsub '
                + '-N %s '%PAR.TITLE
                + '-o %s '%(PATH.SUBMIT +'/'+ 'output.log')
                + '-l %s '%resources
                + '-j %s '%'oe'
                + findpath('system') +'/'+ 'wrappers/submit '
                + '-F %s '%PATH.OUTPUT)

    def run(self, classname, funcname, hosts='all', **kwargs):
        """  Runs tasks in serial or parallel on specified hosts
        """
        self.checkpoint()
        self.save_kwargs(classname, funcname, kwargs)

        if hosts == 'all':
            # run on all available nodes
            queue = range(PAR.NTASK)
            TASK_ID = 0

            while 1:
                unix.run('pbsdsh '
                        + join(findpath('system'), 'wrappers/export_paths.sh ')
                        + os.getenv('PATH') + ' '
                        + os.getenv('LD_LIBRARY_PATH') + ' '
                        + str(TASK_ID) + ' '
                        + join(findpath('system'), 'wrappers/run_pbsdsh ')
                        + PATH.OUTPUT + ' '
                        + classname + ' '
                        + funcname + ' '
                        + dirname(findpath('seisflows')))

                del queue[:PAR.NPROCMAX]
                TASK_ID += 1

                if not queue:
                    break

        elif hosts == 'head':
            # run on head node
            TASK_ID = 0
            unix.run('pbsdsh '
                    + join(findpath('system'), 'wrappers/export_paths.sh ')
                    + os.getenv('PATH') + ' '
                    + os.getenv('LD_LIBRARY_PATH') + ' '
                    + str(TASK_ID) + ' '
                    + join(findpath('system'), 'wrappers/run_pbsdsh_head ')
                    + PATH.OUTPUT + ' '
                    + classname + ' '
                    + funcname + ' '
                    + dirname(findpath('seisflows')))

    def getnode(self):
        """ Gets number of running task
        """
        task_id = int(os.getenv('TASK_ID'))
        node_id = int(os.getenv('PBS_VNODENUM'))
        return ((task_id * PAR.NPROCMAX) + node_id)

    def mpiargs(self):
        """ Call code in serial when using pbsdsh (MPI Singleton)
        """
        return ''

    def save_kwargs(self, classname, funcname, kwargs):
        kwargspath = join(PATH.OUTPUT, 'SeisflowsObjects', classname+'_kwargs')
        kwargsfile = join(kwargspath, funcname+'.p')
        unix.mkdir(kwargspath)
        saveobj(kwargsfile, kwargs)
