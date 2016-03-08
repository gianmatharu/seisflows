
from os.path import join
import numpy as np
import obspy

from seisflows.seistools.ewf2d import Par
from seisflows.tools.config import SeisflowsParameters, SeisflowsPaths, \
    ParameterError, custom_import

PAR = SeisflowsParameters()
PATH = SeisflowsPaths()


class ewf2d(custom_import('preprocess', 'base')):
    """ Data preprocessing class
    """

    def check(self):

        super(ewf2d, self).check()

        if 'DAMPING' not in PAR:
            setattr(PAR, 'DAMPING', 0.0)

        if 'GAIN' not in PAR:
            setattr(PAR, 'GAIN', False)

    def prepare_eval_grad(self, path='.'):
        """ Prepares solver for gradient evaluation by writing residuals and
          adjoint traces
        """
        for channel in self.channels:
            obs = self.reader(path+'/'+'traces/obs', channel)
            syn = self.reader(path+'/'+'traces/syn', channel)

            obs = self.process_traces(obs)
            syn = self.process_traces(syn)

            self.write_residuals(path, syn, obs)
            self.write_adjoint_traces(path+'/'+'traces/adj', syn, obs, channel)


    def evaluate_trial_step(self, path='.', path_try=''):
        """ Prepares solver for gradient evaluation by writing residuals and
            adjoint traces
        """
        for channel in self.channels:
            obs = self.reader(path+'/'+'traces/obs', channel)
            syn = self.reader(path_try, channel)

            obs = self.process_traces(obs)
            syn = self.process_traces(syn)

            self.write_residuals(path_try, syn, obs)


    def process_traces(self, stream):
        """ Performs data processing operations on traces
        """
        nt, dt, _ = self.get_time_scheme(stream)
        n, _ = self.get_network_size(stream)
        df = dt**-1

        for ir in range(n):

            # detrend
            stream[ir].detrend()

            # filter data
            if PAR.FREQLO and PAR.FREQHI:
                stream[ir].filter('bandpass', freqmin=PAR.FREQLO, freqmax=PAR.FREQHI, zerophase=True)
            elif PAR.FREQHI:
                stream[ir].filter('lowpass', freq=PAR.FREQHI, zerophase=True)
            else:
                raise ParameterError(PAR, 'BANDPASS')

        if PAR.DAMPING:
            stream = self.apply_damping(stream)

        if PAR.GAIN:
            stream = self.apply_gain(stream)

        stream = self.convert_to_float(stream)

        return stream


    def write_residuals(self, path, s, d):
        """ Computes residuals from observations and synthetics
        """
        nt, dt, _ = self.get_time_scheme(s)
        n, _ = self.get_network_size(s)

        filename = path +'/'+ 'residuals'
        r = np.zeros(n)

        for i in range(n):
            r[i] = self.misfit(s[i].data, d[i].data, nt, dt)

        np.savetxt(filename, r)


    def write_adjoint_traces(self, path, s, d, channel):
        """ Generates adjoint traces from observed and synthetic traces
        """
        nt, dt, _ = self.get_time_scheme(s)
        n, _ = self.get_network_size(s)

        for i in range(n):
            s[i].data = self.adjoint(s[i].data, d[i].data, nt, dt)

        # normalize traces
        if PAR.NORMALIZE:
            for ir in range(n):
                w = np.linalg.norm(d[i], ord=2)
                if w > 0: 
                    s[:,ir] /= w

        self.writer(s, path, channel, tag='adj')


    # additional processing functions
    def apply_gain(self, stream):
        """ Apply offset dependent gain
        """
        nt, dt, _ = self.get_time_scheme(stream)
        n, _ = self.get_network_size(stream)

        for ir in range(n):
            offset = stream[ir].stats.su.trace_header.\
                     distance_from_center_of_the_source_point_to_the_center_of_the_receiver_group
            stream[ir].data *= offset**2

        return stream


    def apply_damping(self, stream):
        """ Apply exponential time damping.
            Implements a crude first detection approach.
        """
        nt, dt, _ = self.get_time_scheme(stream)
        n, _ = self.get_network_size(stream)

        time = np.arange(0, nt*dt, dt)
        gamma = PAR.DAMPING

        for ir in range(n):
            # threshold for first arrival
            threshold = 1e-3 * max(abs(stream[ir].data))

            if len(np.where(abs(stream[ir].data) > threshold)[0]) != 0:
                t0 = time[np.where(abs(stream[ir].data) > threshold)[0][0]]

                iterable = ((lambda t: 1.0 if t < t0 else (np.exp(-gamma*(t-t0))))(item) for item in time)
                damping_filter = np.fromiter(iterable, dtype='float32')
                stream[ir].data = np.multiply(stream[ir].data, damping_filter)
            else:
                pass

        return stream


    ### utility functions

    def convert_to_float(self, stream):
        """ Converts data to float. Required for SU file formats.
        """

        nt, dt, _ = self.get_time_scheme(stream)
        n, _ = self.get_network_size(stream)

        for i in range(n):
            stream[i].data = stream[i].data.astype(dtype='float32')

        return stream


    def write_zero_traces(self, path, channel):
        #TODO FIX implementation
        from obspy.core.stream import Stream
        from obspy.core.trace import Trace

        # Solver parameter class
        p = Par()
        p.read_par_file(join(PATH.SOLVER_INPUT, 'par_template.cfg'))

        # construct seismic data and headers
        t = Trace(data=np.zeros(p.ntimesteps, dtype='float32'))
        t.stats.delta = p.dt
        s = Stream(t)*PAR.NREC

        # write to disk
        self.writer(s, path, channel)


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



