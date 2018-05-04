from __future__ import division
import numpy as np
from scipy.io import loadmat

from gtfblib.gtfb import ERBspacing_given_N
from gtfblib.ozgtfb import OZGTFB

from test_aux_functions import peak_error

refData = loadmat('test/OZGTFB16kTestdata.mat', squeeze_me=True)
fs = 16000
insig = loadmat('test/Inputdata.mat', squeeze_me=True)['indata16k']
refout = refData['y']

def test_OZGTFB_setup():
    fbOZ = OZGTFB(fs=fs, cfs=ERBspacing_given_N(80, 5000, 32))
    assert(peak_error(fbOZ._feedback, refData['feedback'])<1e-4)
    assert(peak_error(fbOZ._gain, refData['gain'])<1e-10)

def test_OZGTFB_process():
    fbOZ = OZGTFB(fs=fs, cfs=ERBspacing_given_N(80, 5000, 32))
    outsig = fbOZ.process(insig)
    assert(peak_error(outsig, refout)<1e-10)

def test_OZGTFB_process_single():
    fbOZ = OZGTFB(fs=fs, cfs=ERBspacing_given_N(80, 5000, 32))
    outsig = fbOZ.process_single(insig, 10)
    assert(peak_error(outsig, refout[10, :])<1e-10)

def test_OZGTFB_process_memory():
    fbOZ = OZGTFB(fs=fs, cfs=ERBspacing_given_N(80, 5000, 32))
    outsig1 = fbOZ.process(insig[:800])
    outsig2 = fbOZ.process(insig[800:])
    outsig = np.hstack((outsig1, outsig2))
    assert(peak_error(outsig, refout)<1e-10)

def test_OZGTFB_clear():
    fbOZ = OZGTFB(fs=fs, cfs=ERBspacing_given_N(80, 5000, 32))
    _ = fbOZ.process(np.random.randn(1000))
    fbOZ._clear()
    outsig = fbOZ.process(insig)
    assert(peak_error(outsig, refout)<1e-10)
