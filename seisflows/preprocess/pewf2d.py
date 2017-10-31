
import sys
from os.path import join

import numpy as np
from obspy.core.trace import Trace

from seisflows.tools import msg, unix
from seisflows.tools.tools import exists, getset
from seisflows.config import ParameterError
from seisflows.plugins import adjoint, misfit, readers, writers
from seisflows.tools import signal
from seisflows.tools.susignal import swindow, smute_offset, sgain_offset, sdamping

PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']


class pewf2d(object):
    """ Data preprocessing class
    """

    def check(self):

        if 'MISFIT' not in PAR:
            setattr(PAR, 'MISFIT', 'Waveform')

        if 'CHANNELS' not in PAR:
            raise ParameterError(PAR, 'CHANNELS')

        if 'FORMAT' not in PAR:
            setattr(PAR, 'FORMAT', 'su_pewf2d')

        if 'NORMALIZE' not in PAR:
          setattr(PAR, 'NORMALIZE', True)

        # data mute option
        if 'MUTE' not in PAR:
          setattr(PAR, 'MUTE', None)

        # data filtering option
        if 'FILTER' not in PAR:
          setattr(PAR, 'FILTER', False)

        if 'PREFILTER' not in PAR:
            setattr(PAR, 'PREFILTER', False)

        # time damping option
        if 'DAMPING' not in PAR:
            setattr(PAR, 'DAMPING', None)

        # offset gain option
        if 'GAIN' not in PAR:
            setattr(PAR, 'GAIN', None)

        # source time function parameters
        if 'USE_STF_FILE' not in PAR:
            setattr(PAR, 'USE_STF_FILE', False)

        if 'STF_FILE' not in PAR:
            setattr(PAR, 'STF_FILE', 'stf.txt')

        self.check_mute()
        self.check_filter()
        self.check_damping()
        self.check_source()

        # assertions
        if PAR.FORMAT not in dir(readers):
            print msg.ReaderError
            raise ParameterError()

        if PAR.FORMAT not in dir(writers):
            print msg.WriterError
            raise ParameterError()


    def setup(self):
        """ Setup
        """
        # define misfit function and adjoint trace generator
        self.misfit = getattr(misfit, PAR.MISFIT)
        self.adjoint = getattr(adjoint, PAR.MISFIT)

        # define seismic data reader and writer
        self.reader = getattr(readers, PAR.FORMAT)
        self.writer = getattr(writers, PAR.FORMAT)

        # prepare channels list
        self.channels = []
        for char in PAR.CHANNELS:
            self.channels += [char]

        if PAR.USE_STF_FILE:
            self.filter_stf(join(PATH.SOLVER_INPUT, PAR.STF_FILE))


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

            obs = self.process_traces(obs, filter=not PAR.PREFILTER)
            syn = self.process_traces(syn, filter=False)

            self.write_residuals(path, syn, obs)
            self.store_residuals(path, filename, syn, obs)
            self.write_adjoint_traces(path+'/'+'traces/adj', syn, obs, filename)


    def prepare_apply_hess(self, path='.', path_try=''):
        """ Prepares solver for gradient evaluation by writing residuals and
          adjoint traces
        """
        solver = sys.modules['seisflows_solver']

        if exists(path + '/residuals'):
            unix.rm(path + '/residuals')

        for filename in solver.data_filenames:
            obs = self.reader(path+'/'+'traces/obs', filename)
            syn = self.reader(path_try+'/'+'traces/syn', filename)

            obs = self.process_traces(obs, filter=not PAR.PREFILTER)
            syn = self.process_traces(syn, filter=False)

            self.write_adjoint_traces(path_try+'/'+'traces/adj', syn, obs, filename)


    def evaluate_trial_step(self, path='.', path_try=''):
        """ Prepares solver for gradient evaluation by writing residuals and
            adjoint traces
        """
        solver = sys.modules['seisflows_solver']

        if exists(path + '/residuals'):
            unix.rm(path + '/residuals')

        for filename in solver.data_filenames:
            obs = self.reader(path+'/'+'traces/obs', filename)
            syn = self.reader(path_try+'/'+'traces/syn', filename)

            obs = self.process_traces(obs, filter=not PAR.PREFILTER)
            syn = self.process_traces(syn, filter=False)

            self.write_residuals(path_try, syn, obs)


    def process_traces(self, stream, filter=True):
        """ Performs data processing operations on traces
        """
        nt, dt, _ = self.get_time_scheme(stream)
        n, _ = self.get_network_size(stream)
        df = dt**-1

        self.apply_damping(stream)
        self.apply_mute(stream)

        # Filtering
        if filter:
            self.apply_filter(stream)

        return stream


    def write_residuals(self, path, syn, dat):
        """ Computes residuals from observations and synthetics
        """
        nt, dt, _ = self.get_time_scheme(syn)
        nn, _ = self.get_network_size(syn)

        filename = path +'/'+ 'residuals'
        if exists(filename):
            rsd = list(np.loadtxt(filename))
        else:
            rsd = []

        for ii in range(nn):
            rsd.append(self.misfit(syn[ii].data, dat[ii].data, nt, dt))

        np.savetxt(filename, rsd)


    def store_residuals(self, path, filename, s, d):

        nt, dt, _ = self.get_time_scheme(s)
        n, _ = self.get_network_size(s)

        r = s.copy()
        for i in range(n):
            r[i].data[:] = s[i].data[:] - d[i].data[:]

        self.writer(r, path, self._swap_tag(filename, 'res'))


    def sum_residuals(self, files):
        """ Sums squares of residuals

          INPUT
            FILES - files containing residuals
        """
        total_misfit = 0.
        for file in files:
            if PAR.MISFIT == 'Correlation1':
                total_misfit += np.sum(np.loadtxt(file))
            else:
                total_misfit += np.sum(np.loadtxt(file)**2.)
        return total_misfit


    def write_adjoint_traces(self, path, s, d, channel):
        """ Generates adjoint traces from observed and synthetic traces
        """
        nt, dt, _ = self.get_time_scheme(s)
        n, _ = self.get_network_size(s)

        for i in range(n):
            s[i].data = self.adjoint(s[i].data, d[i].data, nt, dt)

            if PAR.CHANNELS == ['p']:
                #print 'Use second derivative'
                s[i].data[1:-1] = (s[i].data[2:] - s[i].data[0:-2])/(2.*dt)
                s[i].data[0] = 0.
                s[i].data[-1] = 0.

        self.writer(s, path, self._swap_tag(channel, 'adj'))


    # signal process

    def apply_gain(self, traces):
        if not PAR.GAIN:
            return traces
        else:
            stream = sgain_offset(traces)


    def apply_damping(self, traces):
        if not PAR.DAMPING:
            return traces
        else:
            stream = sdamping(traces, twin=PAR.DAMPING)


    def apply_mute(self, traces):
        if not PAR.MUTE:
            return traces

        if 'MuteEarlyArrivals' in PAR.MUTE:
            traces = signal.mute_early_arrivals(traces,
                PAR.MUTE_EARLY_ARRIVALS_SLOPE, # (units: time/distance)
                PAR.MUTE_EARLY_ARRIVALS_CONST, # (units: time)
                self.get_time_scheme(traces),
                self.get_source_coords(traces),
                self.get_receiver_coords(traces))

        if 'MuteLateArrivals' in PAR.MUTE:
            traces = signal.mute_late_arrivals(traces,
                PAR.MUTE_LATE_ARRIVALS_SLOPE, # (units: time/distance)
                PAR.MUTE_LATE_ARRIVALS_CONST, # (units: time)
                self.get_time_scheme(traces),
                self.get_source_coords(traces),
                self.get_receiver_coords(traces))

        if 'MuteShortOffsets' in PAR.MUTE:
            traces = signal.mute_short_offsets(traces,
                PAR.MUTE_SHORT_OFFSETS_DIST,
                self.get_source_coords(traces),
                self.get_receiver_coords(traces))

        if 'MuteLongOffsets' in PAR.MUTE:
            traces = signal.mute_long_offsets(traces,
                PAR.MUTE_LONG_OFFSETS_DIST,
                self.get_source_coords(traces),
                self.get_receiver_coords(traces))

        return traces


    def apply_filter(self, traces):
        if not PAR.FILTER:
            return traces

        elif PAR.FILTER == 'Bandpass':
            for tr in traces:
                tr.taper(0.05, type='hann')
                tr.filter('bandpass', freqmin=PAR.FREQMIN, freqmax=PAR.FREQMAX)

        elif PAR.FILTER == 'Lowpass':
            for tr in traces:
                tr.taper(0.05, type='hann')
                tr.filter('lowpass', freq=PAR.FREQ)

        elif PAR.FILTER == 'Highpass':
            for tr in traces:
                tr.taper(0.05, type='hann')
                tr.filter('highpass', freq=PAR.FREQ)

        else:
            raise ParameterError()

        return traces

    ### utility functions

    def convert_to_float(self, stream):
        """ Converts data to float. Required for SU file formats.
        """

        nt, dt, _ = self.get_time_scheme(stream)
        n, _ = self.get_network_size(stream)

        for i in range(n):
            stream[i].data = stream[i].data.astype(dtype='float32')

        return stream


    def get_time_scheme(self, stream):
        # assumes uniformly sampled data across all traces
        dt = stream[0].stats.su.trace_header.\
             sample_interval_in_ms_for_this_trace * 1e-6
        nt = len(stream[0].data)
        t0 = 0.
        return nt, dt, t0


    def get_network_size(self, stream):
        nrec = len(stream)
        nsrc = 1
        return nrec, nsrc


    def get_receiver_coords(self, stream):
        if PAR.FORMAT in ['SU', 'su', 'su_pewf2d']:

            scalco = stream[0].stats.su.trace_header.\
                scalar_to_be_applied_to_all_coordinates

            rx = []
            ry = []
            for trace in stream:
                rx += [trace.stats.su.trace_header.group_coordinate_x
                       /scalco]
                ry += [trace.stats.su.trace_header.group_coordinate_y
                       / scalco]
            return rx, ry

        else:
             raise NotImplementedError


    def get_source_coords(self, traces):
        if PAR.FORMAT in ['SU', 'su', 'su_pewf2d']:
            scalco = traces[0].stats.su.trace_header.\
                scalar_to_be_applied_to_all_coordinates
            sx = []
            sy = []
            for trace in traces:
                sx += [trace.stats.su.trace_header.source_coordinate_x
                       / scalco]
                sy += [trace.stats.su.trace_header.source_coordinate_y
                       / scalco]
            return sx, sy

        else:
             raise NotImplementedError


    def filter_stf(self, f):
        """ Filter source wavelet
        """
        t, stf, dt = self._get_trace_from_ascii(f)
        tr = Trace(stf)
        tr.stats.delta = dt

        self.apply_filter([tr])

        # write as ascii time series
        with open(join(PATH.SOLVER_INPUT, 'stf_f.txt'), 'w') as f:
            for i in range(len(tr.data)):
                f.write('{:f} {:f}\n'.format(t[i], tr.data[i]))


    # parameter checking
    def check_mute(self):
        """ Checks mute settings
        """
        if not PAR.MUTE:
            return

        assert getset(PAR.MUTE) <= set([
            'MuteEarlyArrivals',
            'MuteLateArrivals',
            'MuteShortOffsets',
            'MuteLongOffsets'])

        if 'MuteEarlyArrivals' in PAR.MUTE:
            assert 'MUTE_EARLY_ARRIVALS_SLOPE' in PAR
            assert 'MUTE_EARLY_ARRIVALS_CONST' in PAR
            assert PAR.MUTE_EARLY_ARRIVALS_SLOPE >= 0.

        if 'MuteLateArrivals' in PAR.MUTE:
            assert 'MUTE_LATE_ARRIVALS_SLOPE' in PAR
            assert 'MUTE_LATE_ARRIVALS_CONST' in PAR
            assert PAR.MUTE_LATE_ARRIVALS_SLOPE >= 0.

        if 'MuteShortOffsets' in PAR.MUTE:
            assert 'MUTE_SHORT_OFFSETS_DIST' in PAR
            assert 0 < PAR.MUTE_SHORT_OFFSETS_DIST

        if 'MuteLongOffsets' in PAR.MUTE:
            assert 'MUTE_LONG_OFFSETS_DIST' in PAR
            assert 0 < PAR.MUTE_SHORT_OFFSETS_DIST

        if 'MuteShortOffsets' not in PAR.MUTE:
            setattr(PAR, 'MUTE_SHORT_OFFSETS_DIST', 0.)

        if 'MuteLongOffsets' not in PAR.MUTE:
            setattr(PAR, 'MUTE_LONG_OFFSETS_DIST', 0.)


    def check_filter(self):
        """ Checks filter settings
        """
        assert getset(PAR.FILTER) < set([
            'Bandpass',
            'Lowpass',
            'Highpass'])

        if PAR.FILTER == 'Bandpass':
            if 'FREQMIN' not in PAR: raise ParameterError('FREQMIN')
            if 'FREQMAX' not in PAR: raise ParameterError('FREQMAX')
            assert 0 < PAR.FREQMIN
            assert PAR.FREQMIN < PAR.FREQMAX
            assert PAR.FREQMAX < np.inf

        elif PAR.FILTER == 'Lowpass':
            if 'FREQ' not in PAR: raise ParameterError('FREQ')
            assert 0 < PAR.FREQ <= np.inf

        elif PAR.FILTER == 'Highpass':
            if 'FREQ' not in PAR: raise ParameterError('FREQ')
            assert 0 <= PAR.FREQ < np.inf


    def check_damping(self):
        if PAR.DAMPING:
            assert PAR.DAMPING > 0.


    def check_source(self):
        """ Check source parameters
        """
        if PAR.USE_STF_FILE:
            if not exists(join(PATH.SOLVER_INPUT, PAR.STF_FILE)):
                raise IOError('Source time function file not found.')

    # internal functions
    def _get_trace_from_ascii(self, filename):
        """ Read ascii time series and return trace.
        """
        ts = np.loadtxt(filename)
        t = ts[:, 0]
        d = ts[:, 1]
        dt = t[1] - t[0]

        return t, d, dt

    def _swap_tag(self, filename, suffix=''):
        channel = filename.split('_')[0]
        return channel + '_' + suffix + '.su'



