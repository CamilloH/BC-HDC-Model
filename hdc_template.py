from network import NetworkTopology
import hdcNetwork
import numpy as np
from hdcAttractorConnectivity import HDCAttractorConnectivity
from parametersHDC import n_hdc, lam_hdc
from scipy.optimize import curve_fit
# import matplotlib.pyplot as plt

def norm_shift(i, j, n, amp, sigma):
    if abs(i - j) < float(n) / 2.0:
        # shortest path between i and j doesn't pass 0
        dist = i - j
    else:
        # shortest path between i and j passes 0
        dist = i - (j - n)
        if i > j:
            dist = (i - n) - j
        elif i < j:
            dist = i + (n - j)
    x = (dist / float(n)) * 2 * np.pi
    s = 2*(sigma**2)
    val = amp * (np.sqrt(2*np.e) / np.sqrt(s))*x*np.exp(-(1/s)*(x**2))
    return val

# generates hdc network instance with parameters as defined in the thesis
def generateHDC(useFittingFunction=False, debug=False):
    # calculate weights400k
    attrConn = HDCAttractorConnectivity(n_hdc, lam_hdc)
    if useFittingFunction:
        def fittingFunction(x, A, B, C):
            return A*np.exp(-B*(x**2)) + C
        numDists = int(n_hdc / 2)
        maxDist = np.pi
        X = np.linspace(0.0, maxDist, numDists)
        Y = attrConn.conns[0:numDists]
        popt, _ = curve_fit(fittingFunction, X, Y)
        # hack: write back to attrConn
        attrConn.conns = [fittingFunction(x, *popt) for x in X]
        '''
        if debug:
            plt.plot(X, [fittingFunction(x, *popt) for x in X], label="Fitting function $Ae^{-Bx^2}+C$")
            plt.plot(X, Y, label="Connectivity from [Zha96]")
            plt.legend()
            plt.show()
        '''
    conns = attrConn.connection

    # initialize hdc
    topo = NetworkTopology()
    hdcNetwork.addHDCAttractor(topo, n_hdc, conns)


    # add shift layers
    # HDC -> shift layers
    connHDCS = lambda i, j : conns(i, j) * 0.5

    # shift layers -> HDC
    # offset in neurons
    offset = 5
    strength = 1.0
    def peak_right(to, fr):
        return strength * conns(to, (fr + offset) % n_hdc)
    def peak_left(to, fr):
        return strength * conns(to, (fr - offset) % n_hdc)
    hdcNetwork.addHDCShiftLayers(topo, n_hdc, connHDCS, lambda i, j : peak_right(i, j) - peak_left(i, j), connHDCS, lambda i, j : peak_left(i, j) - peak_right(i, j))

    # make instance
    hdc = topo.makeInstance()

    # initialize
    hdcNetwork.initializeHDC(hdc, 0.0)
    return hdc