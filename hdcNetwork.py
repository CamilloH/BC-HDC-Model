import numpy as np
import copy

from hdcTargetPeak import targetPeakDefault
from helper import angleDistAbs
from neuron import phi, phi_inv

# connectivityFunction(i, j) = weight between i and j
def addHDCAttractor(networkTopology, n, connectivityFunction):
    networkTopology.addLayer('hdc_attractor', n)
    networkTopology.connectLayers('hdc_attractor', 'hdc_attractor', connectivityFunction)
    networkTopology.vectorizeConnections('hdc_attractor', 'hdc_attractor')
def addHDCShiftLayers(networkTopology, n, connHDCL, connLHDC, connHDCR, connRHDC):
    networkTopology.addLayer('hdc_shift_left', n)
    networkTopology.addLayer('hdc_shift_right', n)
    networkTopology.connectLayers('hdc_attractor', 'hdc_shift_left', connLHDC)
    networkTopology.connectLayers('hdc_shift_left', 'hdc_attractor', connHDCL)
    networkTopology.connectLayers('hdc_attractor', 'hdc_shift_right', connRHDC)
    networkTopology.connectLayers('hdc_shift_right', 'hdc_attractor', connHDCR)
    networkTopology.vectorizeConnections('hdc_attractor', 'hdc_shift_left')
    networkTopology.vectorizeConnections('hdc_shift_left', 'hdc_attractor')
    networkTopology.vectorizeConnections('hdc_attractor', 'hdc_shift_right')
    networkTopology.vectorizeConnections('hdc_shift_right', 'hdc_attractor')

# initializes HDC by applying current corresponding to firing rates 10% of the target function
def initializeHDC(networkInstance, center, debug=False):
    def printDebug(s):
        if debug:
            print(s)
    # compute the target firing rates at the neurons' preferred directions
    n = len(networkInstance.getLayer('hdc_attractor'))
    F = [targetPeakDefault(angleDistAbs(x * (2*np.pi / n), center)) for x in range(n)]
    # compute currents corresponding to 10% of those rates
    U = [phi_inv(x * 0.5) for x in F]

    # timestep: 0.5 ms
    dt = 0.0005
    # stimulus time
    t_stim = 0.05
    # settling time
    t_settle = 0.05 
    # interval time
    t_interval = 0.05
    # stopping condition: total change in t_interval less than eps
    eps = 0.1

    # apply stimulus
    networkInstance.setStimulus('hdc_attractor', lambda i : U[i])

    # simulate for ts timesteps
    printDebug("HDC initialization: stimulus applied")
    for i in np.arange(0.0, t_stim, dt):
        networkInstance.step(dt)
    printDebug("HDC initialization: stimulus removed")

    # remove stimulus
    networkInstance.setStimulus('hdc_attractor', lambda i : 0)

    # simulate for t_settle
    for i in np.arange(0.0, t_settle, dt):
        networkInstance.step(dt)

    # simulate in episodes of ts timesteps until the total change in an episode is less than eps
    delta = 2*eps
    it = 0
    while delta > eps:
        it += 1
        delta = 0
        for _ in np.arange(0.0, t_interval, dt):
            ratesBefore = copy.copy(networkInstance.getLayer('hdc_attractor'))
            networkInstance.step(dt)
            ratesAfter = networkInstance.getLayer('hdc_attractor')
            delta += sum([abs(ratesAfter[i] - ratesBefore[i]) for i in range(n)])
        printDebug("HDC initialization: iteration {} done with total change {}".format(it, delta))
    printDebug("HDC initialization: done.")