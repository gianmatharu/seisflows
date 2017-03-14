
import numpy as np


class isotropic(object):
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


class acoustic(isotropic):
    """ Acoustic class
    """
    parameters = ['vp']

    @classmethod
    def par_map_forward(cls, model):
        cls.check_model_forward(model)
        return model

    @classmethod
    def par_map_inverse(cls, model):
        cls.check_model_inverse(model)
        return model


class shear(isotropic):
    """ Shear wave velocity
    """
    parameters = ['vs']

    @classmethod
    def par_map_forward(cls, model):
        cls.check_model_forward(model)
        return model

    @classmethod
    def par_map_inverse(cls, model):
        cls.check_model_inverse(model)
        return model


class elastic(isotropic):
    """ Isotropic elastic class
    """
    parameters = ['vp', 'vs']

    @classmethod
    def par_map_forward(cls, model):
        cls.check_model_forward(model)
        return model

    @classmethod
    def par_map_inverse(cls, model):
        cls.check_model_inverse(model)
        return model

class lame(isotropic):
    """ Lame parameter (bulk, shear modulii) class
    """
    parameters = ['lambda', 'mu']

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

class impedance(isotropic):
    """ Impedance parameter class
    """
    parameters = ['Ip', 'Is']

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

# class for parameter rescaling


class ParRescaler(object):
    """ Class to perform parameter rescaling.
    """

    def __init__(self, scale):
        self.scale = {}
        for key in scale.keys():
            self.scale[key] = scale[key]

    @classmethod
    def unit_scaling(cls, parameters):
        scale = dict(zip(parameters, len(parameters) * [1.0]))
        return cls(scale)

    @classmethod
    def mean_scaling(cls, model):
        scale = {}
        for key in model.keys():
            scale[key] = np.asarray(model[key]).mean()
        return cls(scale)


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

