from __future__ import division
import numpy as np
from scipy.signal import lfilter, lfiltic

from .gtfb import gtfb

class OZGTFB(gtfb):

    def __init__(self, **kwargs):
        """Initialize OZGTFB coefficients."""
        gtfb.__init__(self, **kwargs)

        # calculate filter coefficients
        z = np.exp(1j * self._omega);
        # feedback coefs
        self._feedback = np.ones((self.nfilt, 3))
        self._feedback[:,1] = -2*np.cos(self._omega)*np.exp(-self._normB)
        self._feedback[:,2] = np.exp(-2*self._normB)

        # per-channel gain
        tSG = np.abs(1 - 2*np.cos(self._omega) * z *
            np.exp(-self._normB) + np.exp(-2*self._normB) * z**2)
        fG = np.abs(tSG/(1-z))
        self._gain = fG*tSG**3

        # set initial conditions
        self._clear()

    def _clear(self):
        """clear initial conditions"""
        self._ics0 = [lfiltic([1, -1], self._feedback[n,:], [])
            for n in range(self.nfilt)]
        self._ics1 = [lfiltic([1,], self._feedback[n,:], [])
            for n in range(self.nfilt)]
        self._ics2 = [lfiltic([1,], self._feedback[n,:], [])
            for n in range(self.nfilt)]
        self._ics3 = [lfiltic([1,], self._feedback[n,:], [])
            for n in range(self.nfilt)]

    def process(self, insig):
        """Process one-dimensional signal, returning an array of signals
        which are the outputs of the filters"""
        out = np.zeros((self.nfilt, insig.shape[0]))
        for n in range(self.nfilt):
            stage1out, self._ics0[n] = lfilter([1, -1],
                self._feedback[n,:], insig, zi=self._ics0[n])
            stage2out, self._ics1[n] = lfilter([1,],
                self._feedback[n,:], stage1out, zi=self._ics1[n])
            stage3out, self._ics2[n] = lfilter([1,],
                self._feedback[n,:], stage2out, zi=self._ics2[n])
            stage4out, self._ics3[n] = lfilter([1,],
                self._feedback[n,:], stage3out, zi=self._ics3[n])
            out[n, :] = self._gain[n]*stage4out
        return out

    def process_single(self, insig, n):
        """Process a signal with just one of the filters of the
        filterbank, without affecting the state."""
        stage1out = lfilter([1, -1], self._feedback[n,:], insig)
        stage2out = lfilter([1,], self._feedback[n,:], stage1out)
        stage3out = lfilter([1,], self._feedback[n,:], stage2out)
        stage4out = lfilter([1,], self._feedback[n,:], stage3out)
        return self._gain[n]*stage4out
