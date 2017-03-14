import numpy as np

from seisflows.tools.math import eigsorted

class pca(object):

    def __init__(self, model):
        self.model = model
        self.parameters = model.keys()
        self.cov = np.zeros((len(model), len(model)))
        self.evals = np.asarray([])
        self.evecs = np.asarray([])

    def set_covariance(self):
        """ Compute PCA (Sieminski et al. 2009).
        """
        for ii, ikey in enumerate(self.parameters):
            for jj, jkey in enumerate(self.parameters):
                self.cov[ii, jj] = np.dot(self.model[ikey], self.model[jkey])

    def set_pc(self):
        """ Compute eigenvectors and values.
        """
        self.evals, self.evecs = eigsorted(self.cov)
        self.evals /= sum(self.evals)

    def get_pc(self, shape):
        """ Get principal component
        """

        n = len(self.model)
        keys = [str(item) for item in range(n)]
        model = {}

        for i, key in enumerate(keys):

            # get ith principal vector
            v = self.evecs[:, i]
            model[key] = np.zeros(shape)

            for j, jkey in enumerate(self.parameters):
                model[key] += v[j] * self.model[jkey]

        return model

    def print_pc(self):
        """ Print summary
        """
        print('\nParameters: {}\n'.format(self.parameters))
        for i in range(len(self.evals)):
            print('Principal kernel {}'.format(i))
            print('\t Eigenvalue:   {}'.format(self.evals[i]))
            print('\t Pr. comp:     {}\n'.format(self.evecs[:, i]))
