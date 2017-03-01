import ConfigParser


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
        self.ntask = 0

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
            self.ntask = int(p["nsrc"])
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
    try:
        with open(filename, 'w') as configfile:
            for key, value in dict.items():
                configfile.write(key + ': ' + str(value) + '\n')
    except IOError:
        print('Error writing cfg file.')


def get_cfg_value(filename, key):
    """ Return value for corresponding key in cfg file.
    """
    cfg = read_cfg_file(filename)
    return cfg[key]


# misc
def iter_dirname(n):
    """ return string with iteration directory name
    """
    return 'ITER_{:03d}'.format(n)


def event_dirname(n):
    """ return string with event directory name
    """
    return '{:03d}'.format(n)