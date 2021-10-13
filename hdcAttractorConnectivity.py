import numpy as np
from scipy.fftpack import fft, ifft
from hdcTargetPeak import targetPeakDefault

from neuron import phi, phi_inv, tau
from helper import angleDistAbs

class HDCAttractorConnectivity:
    # n: number of neurons
    # lam: lambda from Zhang 1995 paper
    # f: target activity peak function, None for default
    def __init__(self, n, lam, f=None):
        # initialize and compute F
        self.n = n
        if f == None:
            f = targetPeakDefault
        self.f = f
        F = [f(i * (2*np.pi / n)) for i in range(n)]
        # compute W and store
        W = self.generateConnectivity(F, lam)
        self.conns = W
    def generateConnectivity(self, F, lam):
        # re-implemented from Zhang 1995
        # lam corresponds to lambda, the smoothness parameter

        # U directly computed from F
        U = [phi_inv(f) for f in F]

        # compute fourier transforms
        F_ = fft(F)
        U_ = fft(U)

        # compute fourier coefficients of W according to equation
        W_ = []
        for i in range(len(F_)):
            W_.append((U_[i]*F_[i]) / (lam + (abs(F_[i]))**2))

        # inverse fourier to get W
        W = ifft(W_)
        return W
    # connectivity function for HDC attractor
    def connection(self, i, j):
        def resolve_index(i,j,n):
            if abs(i-j)>float(n)/2.0:
                return abs(abs(i-j)-float(n))
            else:
                return abs(i-j)
        return np.real(self.conns[int(resolve_index(i, j, self.n))])