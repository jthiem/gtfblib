from __future__ import division
import numpy as np
from scipy.signal import lfilter, lfiltic

from .gtfb import gtfb, Hz2ERBnum

class Chen(gtfb):

    def __init__(self, N_GD=None, **kwargs):
        """Initialize coefficients for GTFB as in Chen & Hohmann 2015."""
        gtfb.__init__(self, **kwargs)

        # if N_GD is not set, assume value corresponding to 16 ms
        if N_GD is None:
            N_GD = np.ceil(0.016*self.fs).astype(int)

        # filter coefficient parameter and powers
        self.Ak = np.exp(-self._normB + 1j*self._omega)
        self.Ak2 = self.Ak**2
        self.Ak3 = self.Ak**3
        # per-channel gain is based on the difference (in ERB) to the
        # next filter.  This does not work for the last filter, so we
        # duplicate it
        ERBstep = np.diff(Hz2ERBnum(self.cfs))
        ERBstep = np.hstack((ERBstep, ERBstep[-1]))
        self.Bk = (np.sqrt(2) * ERBstep * ((1-np.abs(self.Ak))**4)
                   /(np.abs(self.Ak)+4*np.abs(self.Ak2)+np.abs(self.Ak3)))
        # phase alignment
        N_PE = np.round(3/self._normB).astype(int)
        self.Ck = np.exp(-1j*self._omega*np.fmin(N_GD, N_PE))
        # group delay
        self.Dk = np.fmax(0, N_GD-N_PE).astype(int)

        # actual coefficients
        # self.ComplexA = [1 -4*C2(i) 6*C2(i)^2 -4*C2(i)^3 C2(i)^4]
        # self.ComplexB = normalizedGain(i)*exp(compensatedPhase(i))*...
        #     [zeros(1,max([0,groupDelayL-envelopPeakN(i)])) ...
        #     0 C2(i) 4*C2(i)^2 C2(i)^3 ...
        #     zeros(1,groupDelayL-max([0,groupDelayL-envelopPeakN(i)]))];
        self.ComplexA = np.zeros((self.nfilt, 5), dtype=np.complex128)
        self.ComplexB = np.zeros((self.nfilt, 4+N_GD), dtype=np.complex128)
        for n in range(self.nfilt):
            self.ComplexA[n, :] = [1, -4*self.Ak[n], 6*self.Ak2[n],
                                   -4*self.Ak3[n], self.Ak[n]**4]
            self.ComplexB[n, self.Dk[n]+1:self.Dk[n]+4] = self.Bk[n] \
                * self.Ck[n] * np.array([self.Ak[n], 4*self.Ak2[n], self.Ak3[n]])

        self._clear()

    def _clear(self):
        """clear initial conditions"""
        self._ics = np.vstack([lfiltic(self.ComplexB[n, :],
            self.ComplexA[n, :], np.complex128([])) for n in range(self.nfilt)])

    def process(self, insig):
        """Process one-dimensional signal, returning an array of signals
        which are the outputs of the filters"""
        out = np.zeros((self.nfilt, insig.shape[0]), dtype=np.complex128)
        for n in range(self.nfilt):
            out[n, :], self._ics[n, :] = lfilter(self.ComplexB[n, :],
                self.ComplexA[n, :], insig, zi=self._ics[n, :])
        return out

    def process_single(self, insig, n):
        """Process a signal with just one of the filters of the
        filterbank, without affecting the state."""
        return lfilter(self.ComplexB[n, :], self.ComplexA[n, :], insig)
