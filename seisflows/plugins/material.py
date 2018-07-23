
import numpy as np

class Acoustic(object):
    """ Acoustic Model parameter base class
    """
    parameters = ['vp', 'rho']

    # validation functions
    @classmethod
    def check_model_forward(cls, model):
        _check_model(['vp', 'rho'], model)

    @classmethod
    def check_model_inverse(cls, model):
        _check_model(cls.parameters, model)

    # map velocity to new parametrization
    @classmethod
    def par_map_forward(cls, model):
        return model

    # map custom parametrization to velocity
    @classmethod
    def par_map_inverse(cls, model):
        return model

class Isotropic(object):
    """ Model parameter base class
    """
    parameters = []

    # validation functions
    @classmethod
    def check_model_forward(cls, model):
        _check_model(['vp', 'vs', 'rho'], model)

    @classmethod
    def check_model_inverse(cls, model):
        _check_model(cls.parameters, model)

    # map velocity to new parametrization
    @classmethod
    def par_map_forward(cls, *args, **kwargs):
        raise NotImplementedError

    # map custom parametrization to velocity
    @classmethod
    def par_map_inverse(cls, *args, **kwargs):
        raise NotImplementedError

# isotropic inversion material classes


class Elastic(Isotropic):
    """ Isotropic elastic class
    """
    parameters = ['vp', 'vs', 'rho']

    @classmethod
    def par_map_forward(cls, model):
        cls.check_model_forward(model)
        return model

    @classmethod
    def par_map_inverse(cls, model):
        cls.check_model_inverse(model)
        return model


class Lame(Isotropic):
    """ Lame parameter (bulk, shear modulii) class
    """
    parameters = ['lambda', 'mu', 'rho']

    @classmethod
    def par_map_forward(cls, model):
        cls.check_model_forward(model)

        output = {}

        vp = model['vp']
        vs = model['vs']
        rho = model['rho']

        output['rho'] = rho
        output['lambda'] = rho * (vp**2. - 2.*vs**2.)
        output['mu'] = rho * vs**2.

        return output

    @classmethod
    def par_map_inverse(cls, model):
        cls.check_model_inverse(model)

        output = {}
        rho = model['rho']
        lam = model['lambda']
        mu = model['mu']

        output['rho'] = rho
        output['vp'] = np.sqrt((lam + 2*mu) / rho)
        output['vs'] = np.sqrt(mu / rho)

        return output


class Impedance(Isotropic):
    """ Impedance parameter class
    """
    parameters = ['Ip', 'Is', 'rho']

    @classmethod
    def par_map_forward(cls, model):
        cls.check_model_forward(model)

        output = {}

        vp = model['vp']
        vs = model['vs']
        rho = model['rho']

        output['rho'] = rho
        output['Ip'] = rho * vp
        output['Is'] = rho * vs

        return output

    @classmethod
    def par_map_inverse(cls, model):
        cls.check_model_inverse(model)

        output = {}
        rho = model['rho']
        Ip = model['Ip']
        Is = model['Is']

        output['rho'] = rho
        output['vp'] = Ip / rho
        output['vs'] = Is / rho

        return output


class Slowness(Isotropic):
    """ Slowness parameter class
    """
    parameters = ['pp', 'ps', 'rho']

    @classmethod
    def par_map_forward(cls, model):
        cls.check_model_forward(model)

        output = {}

        vp = model['vp']
        vs = model['vs']
        rho = model['rho']

        output['rho'] = rho
        output['pp'] = 1 / vp
        output['ps'] = 1 / vs

        return output

    @classmethod
    def par_map_inverse(cls, model):
        cls.check_model_inverse(model)

        output = {}
        rho = model['rho']
        pp = model['pp']
        ps = model['ps']

        output['rho'] = rho
        output['vp'] = 1 / pp
        output['vs'] = 1 / ps

        return output

# Parameter non-dimensionalization classes


class MeanRescaler(object):
    """ Perform mean value rescaling
    """
    def __init__(self, model):
        self.scale = {}
        for key in model.keys():
            self.scale[key] = model[key].mean()

    def forward(self, model, parameters=None):
        rmodel = {}
        for key in parameters or model.keys():
            rmodel[key] = model[key] / self.scale[key]

        return rmodel

    def reverse(self, model, parameters=None):
        rmodel = {}
        for key in parameters or model.keys():
            rmodel[key] = model[key] * self.scale[key]

        return rmodel

    def rescale_gradient(self, grad):
        """ Rescale gradient via chain-rule
        """
        rgrad = {}
        for key in grad.keys():
            rgrad[key] = grad[key] * self.scale[key]

        return rgrad

    def print_scale(self):
        for key in self.scale:
            print 'Rescale value for parameter {}: {:.6e}'.format(key, self.scale[key])


class MinMaxRescaler(object):
    """ Perform Min/Max rescaling
    """
    def __init__(self, model):
        self.min, self.max = {}, {}
        for key in model.keys():
            self.min[key] = model[key].min()
            self.max[key] = model[key].max()

            if self.min[key] == self.max[key]:
                raise ValueError('Min/Max are equal. Zero division.')

    def forward(self, model, parameters=None):
        rmodel = {}
        for key in parameters or model.keys():
            rmodel[key] = (model[key] - self.min[key]) / (self.max[key] - self.min[key])

        return rmodel

    def reverse(self, model, parameters=None):
        rmodel = {}
        for key in parameters or model.keys():
            rmodel[key] = model[key] * (self.max[key] - self.min[key]) + self.min[key]

        return rmodel

    def rescale_gradient(self, grad):
        """ Rescale gradient via chain-rule
        """
        rgrad = {}
        for key in grad.keys():
            rgrad[key] = grad[key] * (self.max[key] - self.min[key])

        return rgrad

    def print_scale(self):
        for key in self.min:
            print 'Min/Max values for parameter {}: {:.6e} - {:.6e}'.format(key, self.min[key],
                                                                            self.max[key])


# empirical density scaling relations

def gardeners(model):
    """ Get density model via Gardener's relation
    """
    _check_model(['vp', 'vs'], model)

    vp = model['vp']
    vs = model['vs']

    rho = 310.*vp**0.25

    # check for water layers
    rho_water = 1050.0
    thresh = 1.0e-3
    rho[vs < thresh] = rho_water

    return rho


def _check_model(parameters, model):
    """ Check that input has necessary parameters
    """
    if not isinstance(model, dict):
        raise TypeError('Expected model dictionary, model is type {}'
                        .format(type(model)))
    # check keys
    for par in parameters:
        if par not in model.keys():
            raise KeyError('Model dictionary is missing required parameter {}'.format(par))