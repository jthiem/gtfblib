#!/usr/local/bin/python3
from __future__ import division
import numpy as np
from scipy.signal import lfilter, lfiltic

from .gtfb import gtfb, Hz2ERBnum

class Slaney(gtfb):

    def __init__(self, N_GD=None, EarQ=9.26449, Bfact=1.019, **kwargs):
        """Initialize coefficients for GTFB as in Slaney 1993."""
        gtfb.__init__(self, EarQ=EarQ, Bfact=Bfact, **kwargs)

    def _clear(self):
        """clear initial conditions"""
        pass

    def process(self, insig):
        """Process one-dimensional signal, returning an array of signals
        which are the outputs of the filters"""
        out = np.zeros((self.nfilt, insig.shape[0]), dtype=np.complex128)
        #for n in range(self.nfilt):
        #    out[n, :], self._ics[n, :] = lfilter(self.ComplexB[n, :],
        #        self.ComplexA[n, :], insig, zi=self._ics[n, :])
        return out

    def process_single(self, insig, n):
        """Process a signal with just one of the filters of the
        filterbank, without affecting the state."""
        #return lfilter(self.ComplexB[n, :], self.ComplexA[n, :], insig)
        pass
