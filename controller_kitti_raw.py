from pybullet_environment import PybulletEnvironment
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
from os import listdir
import re
from datetime import datetime
from SimResult import SimResult
import plotting

# file format: pickle file containing (times, avs, directions), i.e. (array of timestamps, array of angular velocities, array of ground-truth directions)
# folder containing input files
basedir_in = "data"
basedir_out = "results/kitti_raw"

# returns true if the input filename should be used for simulation.
# for executing one file just use 'filename == ...', a python regex is used here
# file_predicate = lambda filename : re.match(r"avs_kitti_raw_(.)*\.p", filename)
file_predicate = lambda filename : filename == "avs_chair.p" or filename == "avs_info_building_1.p"

# rad <=> deg factors
r2d = 180.0 / np.pi
d2r = np.pi / 180.0

# plot error and angular velocity
def plotErrAv(simResult, signedErrs=False):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("error (deg)")
    minErr = min([min(simResult.quad_errs_signed), min(simResult.errs_signed)])
    maxErr = max([max(simResult.quad_errs_signed), max(simResult.errs_signed)])
    maxAbsErr = max([abs(minErr), abs(maxErr)])
    ax1.set_ylim(-1.2 * maxAbsErr, 1.2 * maxAbsErr)
    ax1.plot(simResult.times, simResult.errs_signed, label="Error HDC network vs. ground truth", color="tab:blue")
    ax1.plot(simResult.times, simResult.quad_errs_signed, label="Error integration vs. ground truth", color="tab:green")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    minAv = min(simResult.avs)
    maxAv = max(simResult.avs)
    maxAbsAv = max([abs(minAv), abs(maxAv)])
    ax2.set_ylim(-1.2 * maxAbsAv, 1.2 * maxAbsAv)
    ax2.set_ylabel("angular velocity (deg/s)")
    ax2.plot(simResult.times, simResult.avs[1 : len(simResult.avs)], color="tab:orange", label="Angular velocity from IMU")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax1.plot([min(simResult.times), max(simResult.times)], [0.0, 0.0], linestyle="dotted", color="k")
    fig.tight_layout()
    fig.legend()
    plt.show()

# plot only error
def plotErr(simResults, legend=True):
    plt.xlabel("time (s)")
    plt.ylabel("error (deg)")
    maxAbsErr = max([max(simResult.errs) for simResult in simResults])
    maxTime = max([max(simResult.times) for simResult in simResults])
    plt.ylim(-1.2 * maxAbsErr, 1.2 * maxAbsErr)
    plt.xlim(0.0, maxTime)
    for simResult in simResults:
        plt.plot(simResult.times, simResult.errs_signed, label=simResult.label)
        plt.plot([0.0, maxTime], [0.0, 0.0], linestyle="dotted", color="k")
    if legend:
        plt.legend()
    plt.show()

'''
def plotResult(simResult):

    # plot only angular velocity
    plt.xlabel("time (s)")
    plt.ylabel("angular velocity (deg/s)")
    plt.ylim(-50, 50)
    plt.xlim(0.0, t_episode)
    plt.plot(X, [x * r2d for x in avs])
    plt.plot([0.0, t_episode], [0.0, 0.0], linestyle="dotted", color="k")
    plt.show()

    # plot total rotation
    totalMovements = [0.0] * len(thetas)
    for i in range(1, len(avs)):
        totalMovements[i] = totalMovements[i-1] + abs(r2d * thetas[i - 1])
    plt.plot(X, totalMovements)
    plt.xlabel("time (s)")
    plt.ylabel("total rotation (deg)")
    plt.show()

    # plot relative error
    # begin after 20%
    begin_relerror = int(0.2 * len(X))
    plt.plot(X[begin_relerror:len(X)], [100 * errs[i] / totalMovements[i] for i in range(begin_relerror, len(errs))])
    plt.xlabel("time (s)")
    plt.ylabel("relative error (%)")
    plt.show()
'''


# avs[i] between t=times[i] and t=times[i + 1]; directions[i] at t=times[i]
def runSimulation(times, avs, directions, dt_neuron_min, label=""):
    t_episode = times[-1]

    # performance tracking
    netTimes = []
    stepCounterNet = 0
    decodeTimes = []
    netTimes = []
    stepCounterDecode = 0
    t_before = time.time()

    ### result arrays ###
    # hdc network
    errs_signed = []
    errs = []
    decDirections = []

    # numerical quadrature of angular velocity
    # trapezoid rule is used
    # quad_thetas: changes in angle per timestep
    quad_thetas = []
    quad_errs_signed = []
    quad_errs = []
    quad_dirs = []
    quad_dir = 0.0
    #####################


    # init HDC network at 0.0
    hdc = generateHDC()

    # offset all directions by starting direction
    dir_offset = directions[0]

    for i in tqdm(range(1, len(times))):
        t = times[i]
        dt = times[i] - times[i - 1]
        # angular velocity at beginning of timestep
        av = avs[i - 1]
        direc = (directions[i] - dir_offset) % (2*np.pi)
        # set stimuli
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
        # numerical quadrature of angular velocity
        # trapezoid rule
        quad_theta = (0.5 * avs[i-1] + 0.5 * avs[i]) * dt
        quad_thetas.append(quad_theta)
        quad_dir = (quad_dir + quad_theta) % (2 * np.pi)
        quad_dirs.append(quad_dir)
        quad_errs_signed.append(r2d * helper.angleDist(direc, quad_dir))
        quad_errs.append(abs(quad_errs_signed[-1]))

        # set stimuli
        stimL = getStimL(av)
        stimR = getStimR(av)
        hdc.setStimulus('hdc_shift_left', lambda _ : stimL)
        hdc.setStimulus('hdc_shift_right', lambda _ : stimR)

        # simulate network
        timesteps_neuron = int(np.ceil(dt / dt_neuron_min))
        dt_neuron = dt / float(timesteps_neuron)
        beforeStep = time.time()
        stepCounterNet += timesteps_neuron
        # print(dt, dt_neuron, timesteps_neuron)
        hdc.step(dt_neuron, numsteps=timesteps_neuron)
        afterStep = time.time()
        netTimes.append((afterStep - beforeStep) / timesteps_neuron)

        # get rates
        rates_hdc = list(hdc.getLayer('hdc_attractor'))

        # decode direction
        beforeStep = time.time()
        stepCounterDecode += 1
        decodedDir = helper.decodeAttractor(rates_hdc)
        decDirections.append(decodedDir)
        err_signed_rad = helper.angleDist(direc, decodedDir)
        errs_signed.append(r2d * err_signed_rad)
        errs.append(abs(r2d * err_signed_rad))
        afterStep = time.time()
        decodeTimes.append(afterStep - beforeStep)

    # final calculations
    t_total = time.time() - t_before
    X = times[1:len(times)]

    # print results
    print("############### Begin Simulation results ###############")
    # performance tracking
    print("Average step time network:  {:.4f} ms; {} it/s possible".format(1000.0 * np.mean(netTimes), 1.0/np.mean(netTimes)))
    print("Average time decoding:      {:.4f} ms; {} it/s possible".format(1000.0 * np.mean(decodeTimes), 1.0/np.mean(decodeTimes)))
    print("Steps done network:  {}; Time: {:.3f} s; {:.2f}% of total time".format(stepCounterNet, stepCounterNet * np.mean(netTimes), 100 * stepCounterNet * np.mean(netTimes) / t_total))
    print("Steps done decoding: {}; Time: {:.3f} s; {:.2f}% of total time".format(len(X), len(X) * np.mean(decodeTimes), 100 * len(X) * np.mean(decodeTimes) / t_total))
    print("maximum angular velocity: {:.4f} deg/s".format(max(avs) * r2d))
    print("average angular velocity: {:.4f} deg/s".format(sum([r2d * (x / len(avs)) for x in avs])))
    print("median angular velocity:  {:.4f} deg/s".format(np.median(avs)))
    print("maximum error: {:.4f} deg".format(max(errs)))
    print("average error: {:.4f} deg".format(np.mean(errs)))
    print("median error:  {:.4f} deg".format(np.median(errs)))
    print("################ End Simulation results ################")

    result = SimResult(label, X, [x * r2d for x in avs], directions, decDirections, errs_signed, errs, quad_thetas, quad_dirs, quad_errs_signed, quad_errs)
    return result

all_filenames = listdir(basedir_in)
filenames = list(filter(file_predicate, all_filenames))
filenames.sort()
results = []
for i, filename in enumerate(filenames):
    print("Running simulation {}/{}: {}...".format(i + 1, len(filenames), filename))
    in_file = open("{}/{}".format(basedir_in, filename), "rb")
    in_data = pickle.load(in_file, encoding="bytes")
    (times, avs, directions) = in_data
    result = runSimulation(times, avs, directions, 0.012, label=filename)
    results.append(result)
    plotErrAv(result)
    in_file.close()

plotting.plotErr(results, legend=False)
tnow = datetime.now()
timestamp = "{}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}".format(tnow.year, tnow.month, tnow.day, tnow.hour, tnow.minute, tnow.second)
# outfile = "{}/kitti_raw_results_{}".format(basedir_out, timestamp)
outfile = "{}/private_scenarios_100Hz.p".format(basedir_out)
with open(outfile, 'wb') as fl:
    fl.write(pickle.dumps(results))
    fl.close()