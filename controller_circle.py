import time
import numpy as np
import math
from tqdm import tqdm
import hdcNetwork
from hdcAttractorConnectivity import HDCAttractorConnectivity
from network import NetworkTopology
import matplotlib.pyplot as plt
from polarPlotter import PolarPlotter
import helper
from parametersHDC import n_hdc, weight_av_stim
from hdc_template import generateHDC
from scipy.stats import pearsonr
import random
import pickle
import sys

dt_neuron = 0.0005
steps_at_once = 100

# time in seconds, angular velocity in deg
def simulateCircle(t, av):
    hdc = generateHDC()
    av = ((2*np.pi) / 360) * av
    def getStimL(ahv):
        if ahv < 0.0:
            return 0.0
        else:
            return ahv * weight_av_stim
    def getStimR(ahv):
        if ahv > 0.0:
            return 0.0
        else:
            return - ahv * weight_av_stim
    stimL = getStimL(av)
    stimR = getStimR(av)

    decDirs = [0.0]
    realDirs = [0.0]
    realDir = 0.0
    errors = [0.0]
    times = [0.0]
    # simulate network
    hdc.setStimulus('hdc_shift_left', lambda _ : stimL)
    hdc.setStimulus('hdc_shift_right', lambda _ : stimR)
    for i in tqdm(range(round(t / (dt_neuron * steps_at_once)))):
        hdc.step(dt_neuron, numsteps=steps_at_once)
        rates_hdc = list(hdc.getLayer('hdc_attractor'))
        decodedDir = helper.decodeAttractor(rates_hdc)

        realDir = (realDir + av * dt_neuron * steps_at_once)  % (2 * np.pi)
        realDirs.append(realDir)
        decDirs.append(decodedDir)
        errors.append((180 / np.pi) * helper.angleDist(realDir, decodedDir))
        times.append((i + 1) * dt_neuron * steps_at_once)
    return times, realDir, decDirs, errors

t = 100.0
avs = [0.1, 1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
plt.xlabel("time (s)")
plt.ylabel("error (deg)")
for av in avs:
    times, realDirs, decDirs, errors = simulateCircle(t, av)
    plt.plot(times, errors, label="{} deg/s".format(av))
plt.legend()
plt.show()