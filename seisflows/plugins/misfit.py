
# used by the PREPROCESS class and specified by the MISFIT parameter



import numpy as np
from scipy.signal import hilbert as _analytic
from seisflows.tools.susignal import get_wiener_filter, \
    get_wiener_filter_mat, get_weights


def Waveform(syn, obs, nt, dt):
    # waveform difference
    wrsd = syn-obs
    return np.sqrt(np.sum(wrsd*wrsd*dt))


def Envelope(syn, obs, nt, dt, eps=0.05):
    # envelope difference
    # (Yuan et al 2015, eq 9)
    esyn = abs(_analytic(syn))
    eobs = abs(_analytic(obs))
    ersd = esyn-eobs
    return np.sqrt(np.sum(ersd*ersd*dt))


def InstantaneousPhase(syn, obs, nt, dt, eps=0.05):
    # instantaneous phase 
    # from Bozdag et al. 2011

    r = np.real(_analytic(syn))
    i = np.imag(_analytic(syn))
    phi_syn = np.arctan2(i,r)

    r = np.real(_analytic(obs))
    i = np.imag(_analytic(obs))
    phi_obs = np.arctan2(i,r)

    phi_rsd = phi_syn - phi_obs
    return np.sqrt(np.sum(phi_rsd*phi_rsd*dt))


def Traveltime(syn, obs, nt, dt):
    cc = abs(np.convolve(obs, np.flipud(syn)))
    return (np.argmax(cc)-nt+1)*dt


def TraveltimeInexact(syn, obs, nt, dt):
    # much faster but possibly inaccurate
    it = np.argmax(syn)
    jt = np.argmax(obs)
    return (jt-it)*dt


def Amplitude(syn, obs, nt, dt):
    # cross correlation amplitude
    ioff = (np.argmax(cc)-nt+1)*dt
    if ioff <= 0:
        wrsd = syn[ioff:] - obs[:-ioff]
    else:
        wrsd = syn[:-ioff] - obs[ioff:]
    return np.sqrt(np.sum(wrsd*wrsd*dt))


def Envelope2(syn, obs, nt, dt, eps=0.):
    # envelope amplitude ratio
    # (Yuan et al 2015, eq B-1)
    esyn = abs(_analytic(syn))
    eobs = abs(_analytic(obs))
    raise NotImplementedError


def Envelope3(syn, obs, nt, dt, eps=0.):
    # envelope cross-correlation lag
    # (Yuan et al 2015, eqs B-4)
    esyn = abs(_analytic(syn))
    eobs = abs(_analytic(obs))
    return Traveltime(esyn, eobs, nt, dt)


def InstantaneousPhase2(syn, obs, nt, dt, eps=0.):
    esyn = abs(_analytic(syn))
    eobs = abs(_analytic(obs))

    esyn1 = esyn + eps*max(esyn)
    eobs1 = eobs + eps*max(eobs)

    diff = syn/esyn1 - obs/eobs1

    return np.sqrt(np.sum(diff*diff*dt))


def Correlation1(syn, obs, nt, dt):
    # normalized zero-lag cross-correlation
    nfac = np.sqrt(np.sum(obs*obs*dt)) * np.sqrt(np.sum(syn*syn*dt))
    xcorr = np.sum(syn*obs*dt)

    return _div0(-xcorr, nfac, nt)


def WaveformL1(syn, obs, nt, dt):
    # L1 waveform
    wrsd = abs(syn-obs)
    return np.sqrt(np.sum(wrsd*wrsd*dt))

def Adaptive(syn, obs, nt, dt, reverse=False):
    # AWI objective function
    mu = 1e2
    t = np.arange(0, nt*dt, dt)
    T = get_weights(t, sym=True)

    if reverse:
        #w = get_wiener_filter_mat(syn, obs, mu)
        w = get_wiener_filter(syn, obs, mu)
    else:
        w = get_wiener_filter(obs, syn, mu)
        #w = get_wiener_filter_mat(obs, syn, mu)

    return -0.5 * np.sum(T.dot(w) * T.dot(w)) / np.sum(w*w)

def AdaptiveR(syn, obs, nt, dt):
    return Adaptive(syn, obs, nt, dt, reverse=True)

def Displacement(syn, obs, nt, dt):
    return Exception('This function can only used for migration.')

def Velocity(syn, obs, nt, dt):
    return Exception('This function can only used for migration.')

def Acceleration(syn, obs, nt, dt):
    return Exception('This function can only used for migration.')

def _div0(num, den, n):
    if den == 0:
        return 0
    else:
        return num / den