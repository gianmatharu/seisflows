
import numpy as np
from seisflows.tools.array import check_2d, as_ndarrays
import ConfigParser

# Contains all utility functions corresponding to EWF2D solver.


# Solver configuration parameter class.
class Par(object):
    def __init__(self):
        self.npad = 2
        self.nprocx = 0
        self.nprocz = 0
        self.nx = 0
        self.nz = 0
        self.dx = 0.0
        self.dz = 0.0
        self.dt = 0.0
        self.ntimesteps = 0.0
        self.output_interval = 0
        self.use_free_surface = False
        self.use_cpml_left = False
        self.use_cpml_right = False
        self.use_cpml_top = False
        self.use_cpml_bottom = False
        self.ncpml = 0

    def read_par_file(self, filename):

        try:
            open(filename, 'r')
        except IOError:
            print('Cannot open ' + filename)
            exit(-1)
        else:
            cfg = ConfigParser.SafeConfigParser()
            cfg.readfp(FakeSecHead(open(filename)))
            p = dict(cfg.items('asection'))

            # Assign Par values using key value pairs
            self.nprocx = int(p["nprocx"])
            self.nprocz = int(p["nprocz"])
            self.nx = int(p["nx"])
            self.nz = int(p["nz"])
            self.dx = float(p["dx"])
            self.dz = float(p["dx"])
            self.dt = float(p["dt"])
            self.ntimesteps = int(p["ntimesteps"])
            self.output_interval = int(p["output_interval"])
            self.use_free_surface = self.str2bool(p["use_free_surface"])
            self.use_cpml_left = self.str2bool(p["use_cpml_left"])
            self.use_cpml_right = self.str2bool(p["use_cpml_right"])
            self.use_cpml_top = self.str2bool(p["use_cpml_top"])
            self.use_cpml_bottom = self.str2bool(p["use_cpml_bottom"])
            self.ncpml = int(p["ncpml"])
            if self.use_free_surface and self.use_cpml_top:
                self.use_cpml_top = False

    def str2bool(self, str):
        """ Convert string to boolean
        """

        if str in ['true', 'True', 'TRUE', '1']:
            return True
        elif str in ['false', 'False', 'FALSE', '0']:
            return False
        else:
            raise ValueError('String does not match accepted types')


# configuration file routines
class FakeSecHead(object):
    def __init__(self, fp):
        self.fp = fp
        self.sechead = '[asection]\n'

    def readline(self):
        if self.sechead:
            try:
                return self.sechead
            finally:
                self.sechead = None
        else:
            return self.fp.readline()


def read_cfg_file(filename):
    """ Read a libconfig file (similar to python config files but includes no section headers)
    """
    try:
        open(filename, 'r')
    except IOError:
        print('Cannot open ', filename)
    else:
        cfg = ConfigParser.SafeConfigParser()
        cfg.readfp(FakeSecHead(open(filename)))
        return dict(cfg.items('asection'))


def write_cfg_file(filename, dict):
    """ Write a libconfig file from a dictionary. Does not retain order.
    """

    with open(filename, 'w') as configfile:
        for key, value in dict.items():
            configfile.write(key + ': ' + str(value) + '\n')

# misc


def iter_dirname(n):
    """ return string with iteration directory name
    """
    return 'ITER_{:03d}'.format(n)


def event_dirname(n):
    """ return string with event directory name
    """
    return '{:03d}'.format(n)


def extend_pml_velocities(v, nx, nz, ncpml, left=True, right=True, top=True, bottom=True):
    """ Extends edges of grid velocities into CPML layers to ensure 1D - profiles.
    """

    v = v.reshape((nz, nx))

    if left:
        v[:, :ncpml] = v[:, [ncpml]]

    if right:
        v[:, nx-ncpml:] = v[:, [nx - ncpml - 1]]

    if top:
        v[:ncpml, :] = v[ncpml, :]

    if bottom:
        v[nz - ncpml:, :] = v[nz - ncpml - 1, :]

    return v

def inspect_model(Lx, Lz, vp, vs=None, vpvs=1.76, nu=None, f=[1.0]):
    """ Inspect a Vp model and return characteristics about the model.
    Vs model is either supplied or constructed.
    :param model:
    :param f0:
    :return:
    """

    vp = np.asarray(vp)
    if vp.ndim != 2:
        raise TypeError('Check only suitable for 2D models.')

    if vs == None:
        if nu is None:
            vs = vp / vpvs
        else:
            vs = vp / _get_vpvs_from_poisson(nu)
    else:
        vs = np.asarray(vs)
        if vs.ndim != 2:
            raise TypeError('Check only suitable for 2D models.')


    f = np.asarray(f)
    f.sort()

    fmin = f[0]
    fmax = f[-1]

    vpmin = vp.min()
    vpmax = vp.max()

    vsmin = vs.min()
    vsmax = vs.max()

    vsmean = vs.mean()
    vpmean = vp.mean()

    wpmax = vpmean / fmin
    wsmax = vsmean / fmin

    wpmin = [int((vpmean / item)) for item in f]
    wsmin = [int((vsmean / item)) for item in f]

    vpdisp = vpmin / (5 * fmax)
    vsdisp = vsmin / (5 * fmax)

    print('MODEL DIAGNOSTICS')
    print('X dimension (km):    {}'.format(Lx))
    print('Z dimension (km):    {}'.format(Lz))
    print('Vp Min/Max (km/s):      {:.0f} - {:.0f}'.format(vpmin, vpmax))
    print('Vs Min/Max (km/s):      {:.0f} - {:.0f}'.format(vsmin, vsmax))
    print('\n')

    print('FREQUENCY SELECTION CRITERIA')
    print('Freqs (Hz):      {}'.format(f))
    print('P wave min wavelengths (m):      {}'.format(wpmin))
    print('S wave min wavelengths (m):      {}'.format(wsmin))
    print('P wave max wavelengths (m):      {:.0f}'.format(wpmax))
    print('S wave max wavelengths (m):      {:.0f}'.format(wsmax))
    print('P wavelengths propagated:    {:2f}'.format(1000 * Lz / wpmax))
    print('S wavelengths propagated:    {:2f}'.format(1000 * Lz / wsmax))
    print('\n')

    print('STABILITY CRITERIA')
    print('P wave dispersion - dx (m) < {:.0f}'.format(vpdisp))
    print('S wave dispersion - dx (m) < {:.0f}'.format(vsdisp))
    print('P wave Aliasing - dx (m) < {:.0f}'.format(wpmin[-1] / 2))
    print('S wave Aliasing - dx (m) < {:.0f}'.format(wsmin[-1] / 2))
    print('\n')

def _get_vpvs_from_poisson(nu):
    return np.sqrt(((2 * nu - 2) / (2 * nu - 1)))