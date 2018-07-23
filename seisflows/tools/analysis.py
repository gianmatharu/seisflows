from os.path import join

import numpy as np
import matplotlib.pyplot as plt


class OptimStats(object):
    """ Optimization statistics container.

        Class to allow for reading/writing for inversion statistics.

        or

        base/name/X/output.stats/ (for repeated trials)
    """
    def __init__(self, name, stats_path='output.stats', ntrials=1):
        self.name = name
        self.stats_path = stats_path
        self.ntrials = ntrials
        self.stats = {}

    def __getattr__(self, attr):
        """ Shortcut for accessing stats dict
        """
        try:
            return self.stats[attr]
        except:
            raise KeyError('Attribute not found in class or stats dict.')

    def read_all(self, path):
        """ Read all stats
        """
        for stat in ['misfit', 'step_count',    'step_length', 'restarted']:
            self.read_values(path, stat)

    def read_values(self, path, filename):
        """ Read from a text file
        """
        if self.ntrials > 1:
            self.stats[filename + '_table'] = []
            for i in range(self.ntrials):
                self.stats[filename + '_table'] += [self.load(path, filename, i+1)]

            self.stats[filename] = np.mean(self.stats[filename + '_table'], axis=0)
            self.stats[filename + '_std'] = np.std(self.stats[filename + '_table'], axis=0)
        else:
            self.stats[filename] = self.load(path, filename)

    def get_model_error(self, pars, paths):
        """ Return model error
        """
        if self.misfit is None:
            raise ValueError('No iterations detected')
        else:
            for par in pars:

                # parameter key
                tag = '{}_err'.format(par)
                mtrue = np.fromfile(join(paths['true'], '{}.bin'.format(par)),
                                            dtype='float32')

                # average for random trials
                if self.ntrials > 1:
                    self.stats[tag + '_table'] = np.zeros((self.ntrials, len(self.misfit)))

                    for i in range(self.ntrials):
                        for it in range(len(self.misfit)):
                            if it == 0:
                                mest = np.fromfile(join(paths['init'], '{}.bin'.format(par)),
                                                    dtype='float32')
                            else:
                                mest = np.fromfile(join(paths['est'], '{}/m{:02d}'.format(i+1, it+1),
                                                    '{}.bin'.format(par)),
                                                    dtype='float32')

                            self.stats[tag + '_table'][i, it] = model_error(mtrue, mest)

                    self.stats[tag] = np.mean(self.stats[tag + '_table'], axis=0)
                    self.stats[tag + '_std'] = np.std(self.stats[tag + '_table'], axis=0)

                else:
                    self.stats[tag] = np.zeros(len(self.misfit))

                    for i in range(len(self.misfit)):
                        if i == 0:
                            mest = np.fromfile(join(paths['init'], '{}.bin'.format(par)),
                                                    dtype='float32')
                        else:
                            mest = np.fromfile(join(paths['est'], 'm{:02d}'.format(i+1),
                                                    '{}.bin'.format(par)),
                                                    dtype='float32')

                        self.stats[tag][i] = model_error(mtrue, mest)


    def load(self, path, filename, itrial=None):
        """ Load a text file
        """
        if itrial is None:
            return np.loadtxt(join(path, self.name, self.stats_path, filename))
        else:
            return np.loadtxt(join(path, self.name, '{}'.format(itrial), self.stats_path, filename))

    def plot(self, stat):
        if stat in self.stats:
            plt.plot(self.stats[stat])
            plt.xlabel(stat)


def model_error(mtrue, mest):
    return (100./len(mtrue)) * (np.sum(abs(mtrue-mest) / (mtrue)))