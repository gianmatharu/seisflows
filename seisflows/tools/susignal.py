
import numpy as np
from obspy import Stream
from seisflows.tools.math import nextpow2
from seisflows.tools.graphics import _convert_to_array
import matplotlib.pyplot as plt

import scipy.signal as signal


class FixedStream(Stream):
    """ Custom class for Obspy streams of fixed dimensions.
    """
    def __add__(self, other):
        """ perform addition of trace data in two equal size stream objects
        """
        self._validate_input(other)
        for i in range(len(self.traces)):
            self.traces[i].data += other[i].data

        return FixedStream(self.traces)

    def __sub__(self, other):
        """ perform subtraction of trace data in two equal size stream objects
        """
        self._validate_input(other)
        for i in range(len(self.traces)):
            self.traces[i].data -= other[i].data

        return FixedStream(self.traces)

    def __iadd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __isub__(self, other):
        if other == 0:
            return self
        else:
            return self.__sub__(other)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __rsub__(self, other):
        if other == 0:
            return self
        else:
            return self.__sub__(other)

    def _validate_input(self, other):
        """ Validate stream dimensions.
        """
        if not isinstance(other, Stream):
            raise TypeError('Addition only available for stream object.')

        if len(self.traces) != len(other):
            raise ValueError('Stream objects must have equal length.')

        for trace1, trace2 in zip(self.traces, other):
            if len(trace1) != len(trace2):
                raise ValueError('Trace data must be of equal length.')


def saddnoise(stream, snr=10.0, clean=False, verbose=False):
    """ Add Gaussian noise to data. 
    """
    # compute norm of data
    d = _convert_to_array(stream)
    dnorm = np.linalg.norm(d)**2

    # approximate variance for Gaussian noise
    var = (dnorm / 10**(0.1*snr)) / (d.shape[0] * d.shape[1])

    # generate noise array
    noise = np.sqrt(var) * np.random.randn(d.shape[0], d.shape[1])

    if clean:
        nt = len(stream[0].data)
        dt = stream[0].stats.delta
        time = np.arange(0, nt*dt, dt)

    for i, trace in enumerate(stream):
        if clean:
            threshold = 1e-3 * max(abs(trace.data))

            if len(np.where(abs(trace.data) > threshold)[0]) != 0:
                tc = time[np.where(abs(trace.data) > threshold)[0][0]]

            # find window indicies
            ic = int(tc / dt)
            trace.data[ic:] += noise[ic:, i]
        else:
            trace.data += noise[:, i]

    if verbose:
        nnorm = np.linalg.norm(noise)**2
        snr = 10*np.log10(dnorm/nnorm)
        print "SNR [dB] = {}".format(snr)

    return stream


def slowpass(stream, **options):
    """ Wrapper to obspy.signal.filter.lowpass for stream object.
    """

    for trace in stream:
        trace.filter('lowpass', **options)

    return stream


def sbandpass(stream, **options):
    """ Wrapper to Obspy filter. Performs bandpass filtering via obpsy.signal.filter.bandpass
    """

    for trace in stream:
        trace.filter('bandpass', **options)

    return stream


def swindow(stream, t0=0.0, tmin=None, tmax=None, units='samples', mute_interval=False,
             wtype='tukey', **kwargs):
    """ Perform time windowing on a seismic section with uniform sampling.
    """

    # get data dimensions
    nt = len(stream[0].data)

    dt = stream[0].stats.su.trace_header.\
            sample_interval_in_ms_for_this_trace * 1e-6

    # find window indicies
    if units == 'time':
        itmin = int((tmin-t0) / dt)
        itmax = int((tmax-t0) / dt)
    elif units == 'samples':
        itmin = int(tmin)
        itmax = int(tmax)
    else:
        raise ValueError('units should be samples or physical')

    win = window(nt, itmin, itmax, wtype=wtype, **kwargs)

    if mute_interval:
        win = 1 - win

    # window traces
    for trace in stream:
        trace.data[:] *= win

    return stream


def swindowc(stream, t0=0.0, tc=None, twin=None, mute_interval=False,
             wtype='tukey', **kwargs):
    """ Perform time windowing on a seismic section with uniform sampling.
    """

    # get data dimensions
    nt = len(stream[0].data)

    if not tc:
        raise ValueError('Need centre time.')
    if not twin:
        raise ValueError('Need width of time window')

    dt = stream[0].stats.su.trace_header.\
            sample_interval_in_ms_for_this_trace * 1e-6

    # find window indicies
    ic = int(tc-t0 / dt)
    winlen = int(twin / dt)

    if winlen % 2 != 0:
        winlen += 1

    itmin = int(ic - (winlen / 2))
    itmax = int(ic + (winlen / 2))

    win = window(nt, itmin, itmax, wtype=wtype, **kwargs)

    if mute_interval:
        win = 1 - win

    # window traces
    for trace in stream:
        trace.data[:] *= win

    return stream


def sdamping(stream, twin=None):
    """ Apply exponential time damping.
        Implements a crude first detection approach.
    """
    # get data dimensions
    nt = len(stream[0].data)
    dt = stream[0].stats.su.trace_header.\
            sample_interval_in_ms_for_this_trace * 1e-6

    time = np.arange(0, nt*dt, dt)

    if not twin:
        raise ValueError('Need width of time window')

    nwin = int(twin / dt)
    sigma = nwin / (2 * np.sqrt(2 * np.log(2)))

    for trace in stream:

        win = np.zeros(nt)
        # threshold for first arrival
        threshold = 1e-3 * max(abs(trace.data))

        if len(np.where(abs(trace.data) > threshold)[0]) != 0:
            tc = time[np.where(abs(trace.data) > threshold)[0][0]]

            # find window indicies
            ic = int(tc / dt)
            win = np.exp(-0.5 * (((ic - np.asarray(range(nt))) / sigma)**2))

            trace.data[:] *= win
        else:
            pass

    return stream


def gaussian_window(ic, nt):
    win = range(nt) - ic


def smute_offset(stream, offmax, inner_mute=False):
    """ Mute traces beyond a certain offset.
    stream: obspy stream
        Obspy stream object
    offmax:
        Maximum offset
    """

    if inner_mute:
        def f(x): return abs(x) <= abs(offmax)
    else:
        def f(x): return abs(x) >= abs(offmax)


    offsets = [item.stats.su.trace_header.
               distance_from_center_of_the_source_point_to_the_center_of_the_receiver_group
               for item in stream]

    offsets = np.asarray(offsets)

    # window traces
    for ir, trace in enumerate(stream):

        if f(offsets[ir]):
            trace.data[:] *= 0

    return stream


def sgain_offset(stream):
    """ Mute traces beyond a certain offset.
    stream: obspy stream
        Obspy stream object
    offmax:
        Maximum offset
    """
    offsets = [item.stats.su.trace_header.
               distance_from_center_of_the_source_point_to_the_center_of_the_receiver_group
               for item in stream]

    offsets = np.asarray(offsets)
    offsets = offsets / float(abs(offsets).max())

    weights = offsets**2

    #weights = np.sin(abs(offsets) * 0.5 * np.pi)

    # window traces
    for ir, trace in enumerate(stream):
        trace.data[:] *= offsets[ir]**2

    return stream


def window(nt, imin=None, imax=None, wtype='tukey', **options):
    """ Generate a time window.

    Produces a zero padded window of length nt. Tapered window
    is of length imax-imin. If imin/imax are not provided they
    are set to 0/nt respectively.

    Parameters
    ----------

    nt: int
        length of window
    imin: int
        starting index of window
    imax: int
        end index of window
    type: {'cosine', 'tukey'}
        window type
    kwargs:
        kwargs for window routines

    Returns
    -------

    win: ndarray, ndim=1
        window
    """

    win = np.zeros(nt)
    # set default indicies
    if not imin:
        imin = 0

    if not imax:
        imax = nt

    # bounds check
    if imin > nt or imin < 0:
        raise ValueError('imin out of bounds.')

    if imin > nt or imin < 0:
        raise ValueError('imax out of bounds.')

    if imax <= imin:
        raise ValueError('imin must be less that imax.')

    # generate window
    nw = int(imax - imin)

    try:
        fwin = getattr(signal, wtype)
    except:
        raise AttributeError('Window type not found in scipy.signal')
    else:
        win[imin:imax] = fwin(nw, **options)

    return win


def source_time_function(t, t0, f0, stf_type='ricker', factor=1e10):

    tau = t - t0
    pi_f0_square = np.pi * np.pi * f0 * f0
    tau_square = tau * tau

    src_tf = factor * (1 - 2 * pi_f0_square * tau_square) * np.exp(-pi_f0_square * tau_square)

    return src_tf


def compute_amplitude_spectrum(t, y, Fs, Fmax=None, pad=False, onesided=True, verbose=False):

    if pad:
        n = nextpow2(len(y))
    else:
        n = len(y)

    n = int(n)
    k = np.arange(n)
    freq = (Fs / n) * k

    # Amplitude spectrum
    Y = np.fft.fft(y, n)

    if onesided:
        freq = freq[:int(n/2)]
        Y = Y[:int(n/2)]

    if verbose:
        print('Peak frequency, fmax = {}'.format(freq[abs(Y).argmax()]))
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.xaxis.set_label_text('Time (s)')
        ax1.yaxis.set_label_text('Norm. amp.')
        ax1.set_title('Time series')
        ax1.set_ylim([-1, 1])
        pt = ax1.plot(t, y / abs(y).max())

        ax2 = fig.add_subplot(2, 1, 2)
        ax2.xaxis.set_label_text('Frequency (Hz)')
        ax2.yaxis.set_label_text('Norm. amp.')
        ax2.set_title('Amplitude spectrum')
        ax2.set_ylim([0, 1])
        fi = ax2.fill(freq, abs(Y) / abs(Y).max(), 'c')
        pt = ax2.plot(freq, abs(Y) / abs(Y).max())

        if Fmax:
            ax2.set_xlim([0, Fmax])

        fig.tight_layout()
        plt.show()

    return freq, Y


def get_wiener_filter(d, s, mu):
    """ Obtain a Wiener filter that fits synthetics data in the damped least
    squares sense. || Pw - d || + mu || w ||.
    P is a convolution matrix that performs a temporal convolution between w and s.

    Parameters
    ----------

    d: array_like, ndim=1
        data trace
    s: array_like, ndim=1
        synthetic trace
    mu: float
        damping term to stabilize deconvolution

    Returns
    -------

    w: ndarray, ndim=1
        wiener filter coefficients
    """

    # convert to numpy arrays
    d = np.asarray(d)
    s = np.asarray(s)

    # check size
    if d.ndim > 1 or s.ndim > 1:
        raise ValueError('Only 1-dimensional arrays are supported.')

    # perform convolution/correlation in the frequency domain
    D = np.fft.rfft(d)
    S = np.fft.rfft(s)

    # get correlation/autocorrelation fourier coefficients
    C = D * np.conj(S)
    AC = S * np.conj(S) + mu
    W = C / AC

    # check if real!
    w = np.real(np.fft.irfft(W))

    return w


def get_adjoint_source(T, f, s, w, mu):
    n = len(w)

    #constrct diagonal rescaling matrix
    wnorm = np.linalg.norm(w)
    R = (2 * f * np.eye(n) - T**2) / wnorm
    rw = np.matmul(R, w)

    # deconvolve autocorrelation of predicted from reweighted filter
    W = np.fft.rfft(w)
    P = np.fft.rfft(s)
    RW = np.fft.rfft(rw)
    AC = P * np.conj(P) + mu

    A = RW / AC
    A = A * P
    A = A * np.conj(W)

    adj = np.fft.irfft(A)
    return adj
