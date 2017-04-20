
import sys
from os.path import join

import numpy as np
from obspy.core.trace import Trace
from obspy.core.stream import Stream
from obspy.io.segy.segy import SEGYTraceHeader

from seisflows.tools import msg, unix
from seisflows.tools.tools import exists
from seisflows.tools.susignal import FixedStream
from seisflows.config import ParameterError, custom_import
from seisflows.plugins import adjoint, misfit, readers, writers
from seisflows.plugins.encode import SourceArray
from seisflows.plugins.solver.pewf2d import event_dirname

PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']


class ssewf2d(custom_import('preprocess', 'pewf2d')):
    """ Data preprocessing class
    """

    def check(self):
        # check parameters
        if 'MISFIT' not in PAR:
            setattr(PAR, 'MISFIT', 'Waveform')

        if 'CHANNELS' not in PAR:
            raise ParameterError(PAR, 'CHANNELS')

        if 'READER' not in PAR:
          raise ParameterError(PAR, 'READER')

        if 'WRITER' not in PAR:
          setattr(PAR, 'WRITER', PAR.READER)

        if 'NORMALIZE' not in PAR:
          setattr(PAR, 'NORMALIZE', True)

        # filter settings
        if 'PREFILTER' not in PAR:
            setattr(PAR, 'PREFILTER', False)

        if 'BANDPASS' not in PAR:
          setattr(PAR, 'BANDPASS', False)

        if 'FREQLO' not in PAR:
          setattr(PAR, 'FREQLO', 0.)

        if 'FREQHI' not in PAR:
          setattr(PAR, 'FREQHI', 0.)

        # assertions
        if PAR.READER != 'su_pewf2d':
            raise ValueError('Must use su reader.')

        if PAR.WRITER != 'su_pewf2d':
            raise ValueError('Must use SU writer.')

        if PAR.READER != PAR.WRITER:
            print msg.DataFormatWarning % (PAR.READER, PAR.WRITER)

        if 'STF_FILE' not in PAR:
            raise ParameterError(PAR, 'STF_FILE')


    def setup(self):

        # define misfit function and adjoint trace generator
        self.misfit = getattr(misfit, PAR.MISFIT)
        self.adjoint = getattr(adjoint, PAR.MISFIT)

        # define seismic data reader and writer
        self.reader = getattr(readers, PAR.READER)
        self.writer = getattr(writers, PAR.WRITER)

        # prepare channels list
        self.channels = []
        for char in PAR.CHANNELS:
            self.channels += [char]

        # filter source wavelet
        stf_file = join(PATH.SOLVER_INPUT, PAR.STF_FILE)
        self.filter_stf(stf_file)

        # create directories
        unix.rm(PATH.SOURCE)
        unix.mkdir(PATH.SOURCE)


    def encode_input(self, itask=0, path='', source_array=[], encoding=[]):
        """ Perform source encoding to generate 'supershot'
        """
        if not isinstance(source_array, SourceArray):
            raise TypeError('Expected SourceArray object.')

        # load processed source wavelet and encode
        self.encode_sources(source_array, encoding, itask)

        # encode data
        for channel in self.channels:
            data_list = []
            for source, code in zip(source_array, encoding):

                # read data and process
                dpath = join(PATH.DATA, event_dirname(int(source.index)))
                obs = self.reader(dpath, channel)
                obs = self.process_traces(obs, filter=not PAR.PREFILTER)
                obs = self.encode_traces(obs, code)

                # append to encoded source list
                data_list += [FixedStream(obs)]

            # sum and write data
            data = sum(data_list)
            self.writer(data, path + '/' + 'traces/obs', channel)


    def prepare_eval_grad(self, path='.'):
        """ Prepares solver for gradient evaluation by writing residuals and
          adjoint traces
        """
        if exists(path + '/residuals'):
            unix.rm(path + '/residuals')

        for channel in self.channels:
            obs = self.reader(path+'/'+'traces/obs', channel)
            syn = self.reader(path+'/'+'traces/syn', channel)

            self.write_residuals(path, syn, obs)
            self.store_residuals(path, channel, syn, obs)
            self.write_adjoint_traces(path+'/'+'traces/adj', syn, obs, channel)


    def evaluate_trial_step(self, path='.', path_try=''):
        """ Evaluates trial step during line search.
        """
        if exists(path + '/residuals'):
            unix.rm(path + '/residuals')

        for channel in self.channels:
            obs = self.reader(path+'/'+'traces/obs', channel)
            syn = self.reader(path_try+'/'+'traces/syn', channel)

            self.write_residuals(path_try, syn, obs)


    ### utility functions

    def encode_sources(self, source_array, encoding, itask):
        """ Encode source wavelet.
        """
        if len(source_array) != len(encoding):
            raise ValueError('Dimensions of group do not match encoding')

        # load processed source wavelet
        ntask = len(source_array)
        _, stf, dt = self._get_trace_from_ascii(join(PATH.SOLVER_INPUT, 'stf_f.txt'))

        # initialize stream
        stream = Stream()

        # set headers
        for i in xrange(ntask):
            stream.append(self.get_source_trace(stf, dt, source_array[i].x, source_array[i].z))

        # encode source time functions
        if PAR.ENCODER == 'random':
            for i in xrange(ntask):
                stream[i].data = stream[i].data * encoding[i]
        elif PAR.ENCODER in {'shift', 'plane_wave'}:
            for i in xrange(ntask):
                stream[i] = self.shift_trace(stream[i], encoding[i], PAR.MAX_SHIFT)

        # write encoded sources
        stream = self.convert_to_float(stream)
        stream.write(join(PATH.SOURCE, 'source_{:03d}.su'.format(itask + 1)), format='SU', byteorder='<')

        return stream


    def encode_traces(self, stream, code):
        """ Perform encoding
        """
        if PAR.ENCODER == 'random':
            for trace in stream:
                trace.data = trace.data * code
        elif PAR.ENCODER in {'shift', 'plane_wave'}:
            for trace in stream:
                trace = self.shift_trace(trace, code, PAR.MAX_SHIFT)

        return stream


    def process_traces(self, stream, filter=True):
        """ Performs data processing operations on traces
        """
        nt, dt, _ = self.get_time_scheme(stream)
        n, _ = self.get_network_size(stream)
        df = dt**-1

        # Filtering
        if filter:
            for trace in stream:
                self.filter_trace(trace)

        return stream


    def filter_trace(self, trace):
        """ Filter obspy trace
        """
        if PAR.FREQLO and PAR.FREQHI:
            trace.filter('bandpass', freqmin=PAR.FREQLO, freqmax=PAR.FREQHI, corners=2, zerophase=True)
        elif PAR.FREQHI:
            trace.filter('lowpass', freq=PAR.FREQHI, corners=2, zerophase=True)
        else:
            pass

        return trace


    def filter_stf(self, file):
        """ Load and filter source wavelet.
        """
        t, stf, dt = self._get_trace_from_ascii(file)
        tr = Trace(stf)
        tr.stats.delta = dt

        stf = self.filter_trace(tr)

        # write as ascii time series
        with open(join(PATH.SOLVER_INPUT, 'stf_f.txt'), 'w') as f:
            for i in range(len(stf.data)):
                f.write('{:f} {:f}\n'.format(t[i], stf.data[i]))


    def _get_trace_from_ascii(self, file):
        """ Read ascii time series and return trace.
        """
        ts = np.loadtxt(file)
        t = ts[:, 0]
        d = ts[:, 1]
        dt = t[1] - t[0]

        return t, d, dt


    def get_source_trace(self, d, dt, sx, sz):
        """ Set source position in trace header.
        """
        tr = Trace(d)
        tr.stats.delta = dt
        tr.stats.su = {}
        tr.stats.su.trace_header = SEGYTraceHeader()
        tr.stats.su.trace_header.source_coordinate_x = int(sx)
        tr.stats.su.trace_header.source_coordinate_y = int(sz)
        tr.stats.su.trace_header.scalar_to_be_applied_to_all_coordinates = 1

        return tr

    def shift_trace(self, trace, shift, max_shift):
        """ Adds time shift to a trace. Extends trace 
        """
        n = len(trace.data)
        dt = trace.stats.delta

        nshift = int(shift / dt)
        npad = int(max_shift / dt)

        #if shift % dt != 0:
        #    print('Warning: Shift not divisible by dt')

        if max_shift < shift:
            max_shift = shift

        # shift using fft
        nfft = n + npad
        shifts = np.exp(-(2*np.pi*1j*np.arange(0, n)*nshift) / nfft)

        F = np.fft.rfft(trace.data, nfft)
        trace.data = np.fft.irfft(F*shifts[:(nfft//2)+1], nfft)

        return trace