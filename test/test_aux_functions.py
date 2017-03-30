from __future__ import division
import numpy as np
from scipy.io import loadmat

from gtfblib.gtfb import Hz2ERBnum, ERBnum2Hz, \
    ERBspacing_given_N, ERBspacing_given_spacing

cfs_May = [  80.00,  109.51,  141.84,  177.26,
            216.07,  258.58,  305.16,  356.18,
            412.09,  473.33,  540.43,  613.93,
            694.47,  782.69,  879.35,  985.25,
           1101.26, 1228.36, 1367.60, 1520.15,
           1687.28, 1870.38, 2070.97, 2290.73,
           2531.49, 2795.26, 3084.24, 3400.82,
           3747.66, 4127.64, 4543.93, 5000.00,]

def test_Hz2ERBnum():
    assert(np.abs(Hz2ERBnum(1000)-15.62)<0.005)

def test_ERBnum2Hz():
    assert(np.abs(ERBnum2Hz(1)-26.00)<0.005)

def test_ERBspacing_given_N():
    cfs_Test = ERBspacing_given_N(80, 5000, 32)
    assert(np.max(np.abs(cfs_May - cfs_Test))<0.005)
    
def test_ERBspacing_given_spacing():
    fs = 8000
    cfs_Chen = loadmat('test/Chen8kTestdata.mat', 
                       variable_names='subbandCF', 
                       squeeze_me=True)['subbandCF']
    cfs_Test = ERBspacing_given_spacing(80, 0.9*fs/2, 0.5)
    assert(np.max(np.abs(cfs_Chen - cfs_Test))<1e-10)
    