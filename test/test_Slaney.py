from __future__ import division
import numpy as np
from scipy.io import loadmat

from gtfblib.gtfb import ERBspacing_given_spacing
from gtfblib.slaney import Slaney

from test_aux_functions import peak_error

def test_Slaney_setup():
    # check that the filter coefs are computed the same way
    # as in Slaneys original code
    refData = loadmat('test/Slaney16kTestdata.mat', squeeze_me=True)
    fbSlaney = Slaney(fs=16000, cfs=refData['cf'])
    assert(peak_error(fbSlaney.feedback, refData['ERBfeedback'])<1e-10)
    assert(peak_error(fbSlaney.forward, refData['ERBforward'])<1e-10)
    assert(peak_error(fbSlaney._gain, refData['gain'])<1e-10)

def test_Slaney_process():
    # check that the resulting impulse response matches.
    # NOTE the threshold is set very high: 5e-4.  The reason is that even
    # with the feedback and feedforward coefficients being very close
    # (to within 1e-21) due to the high order this can still result in a
    # large error
    refData = loadmat('test/Slaney16kTestdata.mat', squeeze_me=True)
    fbSlaney = Slaney(fs=16000, cfs=refData['cf'])
    insig = np.zeros((1024,))
    insig[0] = 1
    outsig = fbSlaney.process(insig)
    assert(peak_error(outsig, refData['y'])<5e-4)

def test_Slaney_process_2():
    # repeat the test as above but load the feedback and feedforward
    # coefficients from the MATLAB code.  This makes the result match.
    insig = np.zeros((1024,))
    insig[0] = 1
    refData = loadmat('test/Slaney16kTestdata.mat', squeeze_me=True)
    fbSlaney = Slaney(fs=16000, cfs=refData['cf'])
    fbSlaney.feedback = refData['ERBfeedback']
    fbSlaney.forward = refData['ERBforward']
    outsig = fbSlaney.process(insig)
    assert(peak_error(outsig, refData['y'])<1e-10)

def test_Slaney_process_single():
    # santity check that process_single works.
    insig = np.zeros((1024,))
    insig[0] = 1
    refData = loadmat('test/Slaney16kTestdata.mat', squeeze_me=True)
    fbSlaney = Slaney(fs=16000, cfs=refData['cf'])
    outsig = fbSlaney.process_single(insig, 10)
    assert(peak_error(outsig, refData['y'][10,:])<1e-10)

def test_Slaney_process_memory():
    # check if processing a whole is the same as processing 2 chunks
    insig = np.zeros((1024,))
    insig[0] = 1

    fbSlaney = Slaney(fs=16000)
    refout = fbSlaney.process(insig)

    fbSlaney = Slaney(fs=16000)
    outsig1 = fbSlaney.process(insig[:800])
    outsig2 = fbSlaney.process(insig[800:])
    outsig = np.hstack((outsig1, outsig2))

    assert(peak_error(outsig, refout)<1e-10)

def test_Slaney_clear():
    # make sure that if _clear() is called result is same as a brand
    # new filterbank
    insig = np.zeros((1024,))
    insig[0] = 1

    fbSlaney = Slaney(fs=16000)
    refout = fbSlaney.process(insig)

    fbSlaney = Slaney(fs=16000)
    _ = fbSlaney.process(np.random.randn(1000))
    fbSlaney._clear()
    outsig = fbSlaney.process(insig)
    assert(peak_error(outsig, refout)<1e-10)
