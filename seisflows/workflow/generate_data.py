import system

from os.path import join
from seisflows.tools.config import SeisflowsParameters, SeisflowsPaths, ParameterError
from seisflows.tools.code import exists
from seisflows.tools import unix

PAR = SeisflowsParameters()
PATH = SeisflowsPaths()


class generate_data(object):
    """ Generates synthetic data.
    """

    def check(self):
        """ Checks parameters and paths
        """
        # check paths
        if 'DATA' not in PATH:
            setattr(PATH, 'DATA', None)
    #     setattr(PATH, 'DATA', join(PATH.SUBMIT, 'data'))

        if 'MODEL_TRUE' not in PATH:
            raise ParameterError(PATH, 'MODEL_TRUE')

    def main(self):
        """ Generates data
        """

        # clean directories
        self.clean_directory(PATH.OUTPUT)
        self.clean_directory(PATH.GLOBAL)

        print('Generating data...')
        #unix.mkdir(PATH.DATA)
        system.run('solver', 'setup',
                   hosts='all')
        print('Finished')

    def clean_directory(self, path):
        """ If dir exists clean otherwise make
        """

        if not exists(path):
            unix.mkdir(path)
        else:
            unix.rm(path)
            unix.mkdir(path)