
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

    def setup(self):
        """" Setup
        """
        super(ssewf2d, self).setup()

        # create directories
        unix.rm(PATH.SOURCE)
        unix.mkdir(PATH.SOURCE)


    def encode_input(self, itask=0, path='', source_array=[], encoding=[]):
        """ Perform source encoding to generate 'supershot'
        """
        solver = sys.modules['seisflows_solver']

        if not isinstance(source_array, SourceArray):
            raise TypeError('Expected SourceArray object.')

        # load processed source wavelet and encode
        self.encode_sources(source_array, encoding, itask)

        # encode data
        for filename in solver.data_filenames:
            data_list = []
            for source, code in zip(source_array, encoding):

                # read data and process
                obs = self.reader(join(PATH.DATA, event_dirname(int(source.index))), filename)
                obs = self.process_traces(obs, filter=not PAR.PREFILTER)
                obs = self.encode_traces(obs, code)

                # append to encoded source list
                data_list += [FixedStream(obs)]

            # sum and write data
            data = sum(data_list)
            self.writer(data, path + '/' + 'traces/obs', filename)


    def prepare_eval_grad(self, path='.'):
        """ Prepares solver for gradient evaluation by writing residuals and
          adjoint traces
        """
        solver = sys.modules['seisflows_solver']

        if exists(path + '/residuals'):
            unix.rm(path + '/residuals')

        for filename in solver.data_filenames:
            obs = self.reader(path+'/'+'traces/obs', filename)
            syn = self.reader(path+'/'+'traces/syn', filename)

            self.write_residuals(path, syn, obs)
            self.store_residuals(path, filename, syn, obs)
            self.write_adjoint_traces(path+'/'+'traces/adj', syn, obs, filename)


    def evaluate_trial_step(self, path='.', path_try=''):
        """ Evaluates trial step during line search.
        """
        solver = sys.modules['seisflows_solver']

        if exists(path + '/residuals'):
            unix.rm(path + '/residuals')

        for filename in solver.data_filenames:
            obs = self.reader(path + '/' + 'traces/obs', filename)
            syn = self.reader(path_try + '/' + 'traces/syn', filename)

            self.write_residuals(path_try, syn, obs)


    ### source encoding functions

    def encode_sources(self, source_array, encoding, itask):
        """ Encode source wavelet.
        """
        if len(source_array) != len(encoding):
            raise ValueError('Dimensions of group do not match encoding')

        # load processed source wavelet
        _, stf, dt = self._get_trace_from_ascii(join(PATH.SOLVER_INPUT, 'stf_f.txt'))

        # initialize stream
        stream = Stream()

        # set headers
        for i in xrange(len(source_array)):
            stream.append(self.get_source_trace(stf, dt, source_array[i].x, source_array[i].z))

        # encode source time functions
        if PAR.ENCODER == 'random':
            for i in xrange(len(source_array)):
                stream[i].data = stream[i].data * encoding[i]
        elif PAR.ENCODER in {'shift', 'plane_wave'}:
            for i in xrange(len(source_array)):
                stream[i] = self.shift_trace(stream[i], encoding[i], PAR.MAX_SHIFT)

        # write encoded sources
        self.writer(stream, PATH.SOURCE, 'source_{:03d}.su'.format(itask + 1))
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
            self.apply_filter(stream)

        return stream

    # utility functions

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

        if max_shift < shift:
            max_shift = shift

        # shift using fft
        nfft = n + npad
        shifts = np.exp(-(2*np.pi*1j*np.arange(0, n)*nshift) / nfft)

        F = np.fft.rfft(trace.data, nfft)
        trace.data = np.fft.irfft(F*shifts[:(nfft//2)+1], nfft)

        return trace