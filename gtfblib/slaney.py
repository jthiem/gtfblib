from __future__ import division
import numpy as np
from numpy import cos, exp, pi, sin, sqrt
from scipy.signal import lfilter, lfiltic

from .gtfb import gtfb

class Slaney(gtfb):

    def __init__(self, EarQ=9.26449, Bfact=1.019, **kwargs):
        """Initialize coefficients for GTFB as in Slaney 1993."""
        gtfb.__init__(self, EarQ=EarQ, Bfact=Bfact, **kwargs)
        T = 1/self.fs
        B = self._normB*self.fs
        cf = self.cfs
        # the following copied almost verbatim from Slaneys Apple TR #35
        gain = np.abs((-2*exp(4j*cf*pi*T)*T +
            2*exp(-(B*T) + 2j*cf*pi*T)*T *
            (cos(2*cf*pi*T) - sqrt(3 - 2**(3/2))*
            sin(2*cf*pi*T))) *
            (-2*exp(4j*cf*pi*T)*T +
            2*exp(-(B*T) + 2j*cf*pi*T)*T *
            (cos(2*cf*pi*T) + sqrt(3 - 2**(3/2)) *
            sin(2*cf*pi*T))) *
            (-2*exp(4j*cf*pi*T)*T +
            2*exp(-(B*T) + 2j*cf*pi*T) * T *
            (cos(2*cf*pi*T) -
            sqrt(3 + 2**(3/2))*sin(2*cf*pi*T))) *
            (-2*exp(4j*cf*pi*T)*T+2*exp(-(B*T) + 2j*cf*pi*T) * T *
            (cos(2*cf*pi*T) + sqrt(3 + 2**(3/2))*sin(2*cf*pi*T))) /
            (-2 / exp(2*B*T) - 2*exp(4j*cf*pi*T) +
            2*(1 + exp(4j*cf*pi*T)) / exp(B*T))**4)
        self.feedback = np.zeros((self.nfilt, 9))
        self.forward = np.zeros((self.nfilt, 5))
        self.forward[:,0] = T**4 / gain
        self.forward[:,1] = -4*T**4*cos(2*cf*pi*T)/exp(B*T)/gain
        self.forward[:,2] = 6*T**4*cos(4*cf*pi*T)/exp(2*B*T)/gain
        self.forward[:,3] = -4*T**4*cos(6*cf*pi*T)/exp(3*B*T)/gain
        self.forward[:,4] = T**4*cos(8*cf*pi*T)/exp(4*B*T)/gain
        self.feedback[:,0] = np.ones((self.nfilt,))
        self.feedback[:,1] = -8*cos(2*cf*pi*T)/exp(B*T)
        self.feedback[:,2] = 4*(4 + 3*cos(4*cf*pi*T))/exp(2*B*T)
        self.feedback[:,3] = -8*(6*cos(2*cf*pi*T) + cos(6*cf*pi*T))/exp(3*B*T)
        self.feedback[:,4] = 2*(18 + 16*cos(4*cf*pi*T) + cos(8*cf*pi*T))/exp(4*B*T)
        self.feedback[:,5] = -8*(6*cos(2*cf*pi*T) + cos(6*cf*pi*T))/exp(5*B*T)
        self.feedback[:,6] = 4*(4 + 3*cos(4*cf*pi*T))/exp(6*B*T)
        self.feedback[:,7] = -8*cos(2*cf*pi*T)/exp(7*B*T)
        self.feedback[:,8] = exp(-8*B*T)
        # for testing and debugging
        self._gain = gain
        # last task: initialize memory
        self._clear()

    def _clear(self):
        """clear initial conditions"""
        self._ics = np.vstack([lfiltic(self.forward[n, :],
            self.feedback[n, :], np.zeros((1))) for n in range(self.nfilt)])

    def process(self, insig):
        """Process one-dimensional signal, returning an array of signals
        which are the outputs of the filters"""
        out = np.zeros((self.nfilt, insig.shape[0]))
        for n in range(self.nfilt):
            out[n, :], self._ics[n, :] = lfilter(self.forward[n, :],
                self.feedback[n, :], insig, zi=self._ics[n, :])
        return out

    def process_single(self, insig, n):
        """Process a signal with just one of the filters of the
        filterbank, without affecting the state."""
        return lfilter(self.forward[n, :], self.feedback[n, :], insig)
