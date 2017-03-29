from __future__ import division
import numpy as np
from scipy.io import loadmat

from gtfblib.gtfb import Hz2ERBnum, ERBnum2Hz, \
    ERBspacing_given_N, ERBspacing_given_spacing

def test_Hz2ERBnum():
    assert(np.abs(Hz2ERBnum(1000)-15.62)<0.005)

def test_ERBnum2Hz():
    assert(np.abs(ERBnum2Hz(1)-26.00)<0.005)

def test_ERBspacing_given_N():
    assert(False)
    
def test_ERBspacing_given_spacing():
    fs = 8000
    cfs_Chen = loadmat('test/Chen8kTestdata.mat', 
                       variable_names='subbandCF', 
                       squeeze_me=True)['subbandCF']
    cfs_Test = ERBspacing_given_spacing(80, 0.9*fs/2, 0.5)
    assert(np.sum(np.abs(cfs_Chen - cfs_Test))<0.001)
    