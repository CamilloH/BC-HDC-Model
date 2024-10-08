import math
import numpy as np
import parametersBC as p
import subroutines

def headingCellsActivityTraining(heading):

    sig = 0.25 #0.1885  #higher value means less deviation and makes the boundray rep more accurate
    sig = p.nrHDC * sig / (2 * math.pi)
    amp = 1
    hdRes = 2 * math.pi / 100
    heading_vector = np.repeat(heading, 100)

    tuning_vector = np.linspace(0, 2 * math.pi, p.nrHDC)     # nrHdc = 100  np.arange(0, 2 * math.pi, hdRes)
    # normal gaussian for hdc activity profile
    activity_vector = np.multiply(amp,
                                  (np.exp(-np.power((heading_vector - tuning_vector) / 2 * (math.pow(sig, 2)), 2))))
    return np.around(activity_vector, 5)



