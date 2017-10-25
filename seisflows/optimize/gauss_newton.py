import sys

from seisflows.config import ParameterError, custom_import

PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']


class gauss_newton(custom_import('optimize', 'newton')):
    """ Implements Gauss-Newton-CG algorithm
    """

    def check(cls):
        """ Checks parameters and paths
        """
        super(gauss_newton, cls).check()


    def hessian_product(cls, h):
        return cls.load('g_lcg')/h
