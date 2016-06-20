
import numpy as np
from seisflows.tools.array import check_2d, as_ndarrays
import ConfigParser

# Contains all utility functions corresponding to EWF2D solver.


# Solver configuration parameter class.
class Par(object):
    def __init__(self):
        self.npad = 2
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


def model_diagnosis(p, vp, vs, f, dx, dr):
    """ Inspect a Vp model and return characteristics about the model.
    Vs model is either supplied or constructed.
    """
    if not isinstance(p, Par):
        raise TypeError('p should be of type Par')

    # extract grid interior
    vp, vs = np.asarray(vp), np.asarray(vs)

    if vp.ndim != 2:
        raise TypeError('Check only suitable for 2D models.')

    if vs.ndim != 2:
        raise TypeError('Check only suitable for 2D models.')

    if vp.ndim != 2:
        raise TypeError('Check only suitable for 2D models.')

    if vs.ndim != 2:
        raise TypeError('Check only suitable for 2D models.')

    # Length of model (excluding boundaries)
    nx = p.nx
    nz = p.nz

    Lx = nx * p.dx
    Lz = nz * p.dz

    # Get min/max velocities
    vpmin, vpmax = vp.min(), vp.max()
    vsmin, vsmax = vs.min(), vs.max()
    vpmean, vsmean = vp.mean(), vs.mean()

    wp, ws = (vpmin / f), (vsmin / f)

    vpdisp, vsdisp = (vpmin / (5 * f)), (vsmin / (5 * f))

    # Courant number for fourth order derivative operator
    courant = 7.0 / 6.0
    cfl = p.dx / (courant * np.sqrt(2) * vpmax)

    print('\n')
    print('MODEL DIMENSIONS ------------------------')
    print('Nx, Nz:          ({}, {})'.format(nx, nz))
    print('dx - spacing (m):    {}'.format(p.dx))
    print('X dimension (km):    {}'.format(Lx))
    print('Z dimension (km):    {}'.format(Lz))
    print('\n')

    print('VELOCITY ATTRIBUTES ------------------------')
    print('Vp Min/Max (km/s):      {:.0f} - {:.0f}'.format(vpmin, vpmax))
    print('Vs Min/Max (km/s):      {:.0f} - {:.0f}'.format(vsmin, vsmax))
    print('Vp mean (km/s):      {:.0f}'.format(vpmean))
    print('Vs mean (km/s):      {:.0f}'.format(vsmean))
    print('\n')

    print('FREQUENCY SELECTION CRITERIA ------------------------')
    print('Freqs (Hz):      {}'.format(f))
    print('P wave min wavelengths (m):      {:.0f}'.format(wp))
    print('S wave min wavelengths (m):      {:.0f}'.format(ws))
    print('P Far-field resolution (m):      {:.0f}'.format(wp/2))
    print('S Far-field resolution (m):      {:.0f}'.format(ws/2))
    print('P wavelengths propagated:    {:.2f}'.format(Lz / wp))
    print('S wavelengths propagated:    {:.2f}'.format(Lz / ws))
    print('\n')


    print('SAMPLING CRITERIA ------------------------')
    print('Source spacing (m):        {}'.format(dx))
    print('Receiver spacing (m):        {}'.format(dr))
    print('P aliasing limit (horz) (m):       {:.0f}'.format(wp * 0.5))
    print('S aliasing limit (horz) (m):       {:.0f}'.format(ws * 0.5))
    print('\n')

    print('STABILITY CRITERIA')
    if p.dt < cfl:
        print('CFL condition passes:      dt < dx / (dqrt(2)*C*Vmax) = {:.2e}'.format(cfl))
    else:
        print('CFL condition fails:      dt > dx / (dqrt(2)*C*Vmax) = {:.2e}'.format(cfl))
        print('Current dt:      {:.2e}'.format(p.dt))
    if p.dx < vpdisp:
        print('P wave dispersion (5 ppw):        dx < {:.0f}'.format(vpdisp))
    else:
        print('P wave dispersion expected:       dx  > {:.0f}'.format(vpdisp))
    if p.dx < vsdisp:
        print('S wave dispersion (5 ppw):       dx  < {:.0f}'.format(vsdisp))
    else:
        print('S wave dispersion expected:      dx > {:.0f}'.format(vsdisp))
    print('\n')



def _get_vpvs_from_poisson(nu):
    return np.sqrt(((2 * nu - 2) / (2 * nu - 1)))

