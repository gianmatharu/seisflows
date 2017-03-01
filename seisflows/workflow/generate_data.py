import sys

from seisflows.tools import unix
from seisflows.tools.tools import exists
from seisflows.config import ParameterError

PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']

system = sys.modules['seisflows_system']

class generate_data(object):
    """ Generates synthetic data.
    """


    def check(self):
        """ Checks parameters and paths
        """
        # check paths
        if 'DATA' not in PATH:
            setattr(PATH, 'DATA', None)

        if 'MODEL_TRUE' not in PATH:
            raise ParameterError(PATH, 'MODEL_TRUE')

        # check parameters
        if PAR.SYSTEM != 'serial':
            raise ValueError('Use system class "serial" here.')


    def main(self):
        """ Generates data
        """
        # clean directories
        self.clean_directory(PATH.OUTPUT)
        self.clean_directory(PATH.SCRATCH)

        # generate data
        system.run('solver', 'setup',
                    hosts='all')
        print('Generating data...')
        system.run('solver', 'generate_data',
                    hosts='head')
        print('Finished')

    def clean_directory(self, path):
        """ If dir exists clean otherwise make
        """

        if not exists(path):
            unix.mkdir(path)
        else:
            unix.rm(path)
            unix.mkdir(path)
