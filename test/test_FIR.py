from __future__ import division
import numpy as np
from scipy.io import loadmat

from gtfblib.gtfb import ERBspacing_given_N
from gtfblib.fir import FIR

from test_aux_functions import peak_error

fs = 16000
insig = loadmat('test/Inputdata.mat', squeeze_me=True)['indata16k']
refout = loadmat('test/FIR16kTestdata.mat', squeeze_me=True)['y']

def test_FIR_process():
    fbFIR = FIR(fs=fs, cfs=ERBspacing_given_N(80, 5000, 32))
    outsig = fbFIR.process(insig)
    assert(peak_error(outsig, refout)<1e-10)

def test_FIR_process_single():
    fbFIR = FIR(fs=fs, cfs=ERBspacing_given_N(80, 5000, 32))
    outsig = fbFIR.process_single(insig, 10)
    assert(peak_error(outsig, refout[10, :])<1e-10)

def test_FIR_process_memory():
    fbFIR = FIR(fs=fs, cfs=ERBspacing_given_N(80, 5000, 32))
    outsig1 = fbFIR.process(insig[:800])
    outsig2 = fbFIR.process(insig[800:])
    outsig = np.hstack((outsig1, outsig2))
    assert(peak_error(outsig, refout)<1e-10)

def test_FIR_clear():
    fbFIR = FIR(fs=fs, cfs=ERBspacing_given_N(80, 5000, 32))
    _ = fbFIR.process(np.random.randn(1000))
    fbFIR._clear()
    outsig = fbFIR.process(insig)
    assert(peak_error(outsig, refout)<1e-10)
