import numpy as np

# re-implemented from Zhang 1995

def targetPeak(A, B, K, x):
    return A + B*np.exp(K*np.cos(x))

K = 5.29
A = 1.72
B = 0.344
def targetPeakDefault(x):
    return A + B*np.exp(K*np.cos(x))