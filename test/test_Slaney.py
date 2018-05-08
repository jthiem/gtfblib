from __future__ import division
import numpy as np
from scipy.io import loadmat

from gtfblib.gtfb import ERBspacing_given_spacing
from gtfblib.slaney import Slaney

from test_aux_functions import peak_error

def test_Slaney_setup():
    refData = loadmat('test/Slaney16kTestdata.mat', squeeze_me=True)
    fbSlaney = Slaney(fs=16000, cfs=refData['cf'])
    assert(peak_error(fbSlaney.feedback, refData['ERBfeedback'])<1e-10)
    assert(peak_error(fbSlaney.forward, refData['ERBforward'])<1e-10)

def test_Slaney_process():
    pass
    
def test_Slaney_process_single():
    pass

def test_Slaney_process_memory():
    pass

def test_Slaney_clear():
    pass
