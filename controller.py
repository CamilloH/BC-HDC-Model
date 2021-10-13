import BCActivity

from polarBCplotterAllLayers import BCplotterAllLayers
from polarBCplotter import BCplotter
import BCsimulation
import HDCActivity
from pybullet_environment import PybulletEnvironment
import time
import numpy as np
import math
from tqdm import tqdm
import hdcNetwork
from hdcAttractorConnectivity import HDCAttractorConnectivity
from network import NetworkTopology
import matplotlib.pyplot as plt
import matplotlib
from polarPlotter import PolarPlotter
import helper
import parametersBC
from parametersHDC import n_hdc, weight_av_stim
from hdc_template import generateHDC
from scipy.stats import pearsonr
import random
import pickle
import sys

'''
interactive_param_mode = False
if len(sys.argv) > 1:
    if sys.argv[1] == "interactive":
        interactive_param_mode = True

def queryParam(text, values=None, dtype=None, defaultValue=None, printOptions=True):
    def prompt():
        options = ""
        if printOptions:
            options = "(" + ", ".join(list(values.keys())) + ")"
        default = ", (Default: {})".format(defaultValue) if defaultValue != None else ""
        print("{} {}{}: ".format(text, options, default), end="")
        x = input()
        if x == "" and defaultValue != None:
            return defaultValue
        else:
            return x
    if values != None:
        while True:
            x = prompt()
            if x in values.keys():
                return values[x]
    elif dtype != None:
        while True:
            try:
                x = dtype(prompt())
                return x
            except ValueError:
                pass
    else:
        return None
'''

######## for running from data ########
run_from_data = False
# thetas_file is a pickle file containing a tuple (thetas, t_episode, dt_robot)
# with thetas an array of the change in angle during every timestep
thetas_file = "data/thetas_kitti_raw_2011_09_29.p"
#######################################


######## for running pybullet #########
# total episode time in seconds
t_episode = 180 # 375 = original from HDC 180:curved 240:maze
# minimum neuron model timestep
dt_neuron_min = 0.01
# dt_neuron_min = 0.0005
# robot timestep
dt_robot = 0.05
# simulation environment, available models: "maze", "plus" , "curved", "eastEntry"
env_model = "curved"
# the simulation environment window can be turned off, speeds up the simulation significantly
env_visualize = True
#######################################


###### matplotlib visualization #######
rtplot = True
plotfps = 5.0
#######################################

'''
if interactive_param_mode:
    run_from_data = queryParam("Run from file or run PyBullet simulation?", values = {"F" : True, "S" : False}, defaultValue="S")
    if not run_from_data:
        env_model = queryParam("Select input file", values = {"maze" : "maze", "plus" : "plus"}, defaultValue="maze")
exit(0)
'''

####### noisy angular velocity ########
# The HDC network wasn't found to be less sensitive to any type of noise, thus noise isn't included in interactive parameter selection.
use_noisy_av = False

# gaussian noise
# relative standard deviation (standard deviation = rel. sd * av)
noisy_av_rel_sd = 0.0
# absolute standard deviation (deg)
noisy_av_abs_sd = 0.0

# noise spikes
# average noise spike frequency in Hz
noisy_av_spike_frequency = 0.0
# average magnitude in deg/s
noisy_av_spike_magnitude = 0.0
# standard deviation in deg/s
noisy_av_spike_sd = 0.0

# noise oscillation
noisy_av_osc_frequency = 0.0
noisy_av_osc_magnitude = 0.0
noisy_av_osc_phase = 0.0
#######################################

# Initialize environment
if not run_from_data:
    env = PybulletEnvironment(1/dt_robot, env_visualize, env_model)
    env.reset()
else:
    with open(thetas_file, "rb") as fl:
        (thetas_in, t_episode, dt_robot) = pickle.load(fl)
        fl.close()


# find dt_neuron < dt_neuron_min = dt_robot / timesteps_neuron with timesteps_neuron integer
timesteps_neuron = math.ceil(dt_robot/dt_neuron_min)
dt_neuron = dt_robot / timesteps_neuron

print("neuron timesteps: {}, dt={}".format(timesteps_neuron, dt_neuron))

# init plotter
nextplot = time.time()
# Plots HDC ACtivity, if plotted as well simulation speed will decrease
#if rtplot:
 #   if env_model == "maze":
  #      plotter = PolarPlotter(n_hdc, 0.5 * np.pi, False)
   # else:
    #    plotter = PolarPlotter(n_hdc, 0, False)


##################   BC-Model ###############
plotAllLayers = False
# If you want all Layers plotted the simulation decreases drastically
if plotAllLayers:
    bcPlotter = BCplotterAllLayers()
else:
    bcPlotter = BCplotter()
# variables needed for plotting
eBCSummed = 0
bvcSummed = 0
eBCRates = []
bvcRates = []
eachRateDiff = []
xPositions = []
yPositions = []
sampleX = []
sampleY = []
sampleT = []
bcTimes = []
################## end BC-model ##################
# init HDC
hdc = generateHDC()

realDir = 0.0
avs = []
errs = []
errs_signed = []
thetas = []
netTimes = []
robotTimes = []
plotTimes = []
transferTimes = []
decodeTimes = []

errs_noisy_signed = []
noisyDir = 0.0

t_before = time.time()
t_ctr = 0
for t in tqdm(np.arange(0.0, t_episode, dt_robot)):
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
    def getNoisyTheta(theta):
        noisy_theta = theta
        # gaussian noise
        if noisy_av_rel_sd != 0.0:
            noisy_theta = random.gauss(noisy_theta, noisy_av_rel_sd * theta)
        if noisy_av_abs_sd != 0.0:
            noisy_theta = random.gauss(noisy_theta, noisy_av_abs_sd * dt_robot * (1/r2d))
        # noise spikes
        if noisy_av_spike_frequency != 0.0:
            # simplified, should actually use poisson distribution
            probability = noisy_av_spike_frequency * dt_robot
            if random.random() < probability:
                deviation = random.gauss(noisy_av_spike_magnitude * dt_robot * (1/r2d), noisy_av_spike_sd * dt_robot * (1/r2d))
                print(deviation)
                if random.random() < 0.5:
                    noisy_theta = noisy_theta + deviation
                else:
                    noisy_theta = noisy_theta - deviation
        # noise oscillation
        if noisy_av_osc_magnitude != 0.0:
            noisy_theta += noisy_av_osc_magnitude * dt_robot * (1/r2d) * np.sin(noisy_av_osc_phase + noisy_av_osc_frequency * t)
        return noisy_theta

    # rad to deg factor
    r2d = (360 / (2*np.pi))

    # robot simulation step
    action = []
    beforeStep = time.time()
    if not run_from_data:
        theta = env.step(action)
    else:
        theta = thetas_in[t_ctr]
    afterStep = time.time()
    robotTimes.append((afterStep - beforeStep))
    thetas.append(theta)

    # current is calculated from angular velocity
    angVelocity = theta * (1.0/dt_robot)
    # add noise
    noisy_theta = getNoisyTheta(theta)
    av_net = noisy_theta * (1.0/dt_robot) if use_noisy_av else angVelocity
    avs.append(angVelocity)
    stimL = getStimL(av_net)
    stimR = getStimR(av_net)
    # print(av_net, stimL, stimR)

    # simulate network
    beforeStep = time.time()
    hdc.setStimulus('hdc_shift_left', lambda _ : stimL)
    hdc.setStimulus('hdc_shift_right', lambda _ : stimR)

    hdc.step(dt_neuron, numsteps=timesteps_neuron)
    afterStep = time.time()
    netTimes.append((afterStep - beforeStep) / timesteps_neuron)

    rates_hdc = list(hdc.getLayer('hdc_attractor'))
    rates_sl = list(hdc.getLayer('hdc_shift_left'))
    rates_sr = list(hdc.getLayer('hdc_shift_right'))

    # decode direction, calculate errors
    beforeStep = time.time()
    decodedDir = helper.decodeAttractor(rates_hdc)
    realDir = (realDir + theta) % (2 * np.pi)
    noisyDir = (noisyDir + noisy_theta) % (2 * np.pi)
    err_noisy_signed_rad = helper.angleDist(realDir, noisyDir)
    errs_noisy_signed.append(r2d * err_noisy_signed_rad)
    err_signed_rad = helper.angleDist(realDir, decodedDir)
    errs_signed.append(r2d * err_signed_rad)
    errs.append(abs(r2d * err_signed_rad))
    afterStep = time.time()
    decodeTimes.append(afterStep - beforeStep)


    ################################################    BC-Model    ################################################
    beforeBCStep = time.time()
    raysThatHit = env.getRays()
    polar_angles = np.linspace(0, 2*math.pi - parametersBC.polarAngularResolution, 51)   # - parametersBC.polarAngularResolution

    rayDistances = np.array(raysThatHit)
    rayAngles = np.where(rayDistances == -1, rayDistances, polar_angles)
    '''
    ############### Simulation of only 180° FOV##############
    # Silences everything that is detected behind the agent
    for lk in range(25):
        rayAngles[lk + 13] = -1
        rayDistances[lk + 13] = -1
    '''
    noEntriesRadial = np.where(rayDistances == -1)
    noEntriesAngular = np.where(rayAngles == -1)

    # get boundary points
    thetaBndryPts = np.delete(rayAngles, noEntriesAngular)
    if env_model == "maze" or "curved" or "eastEntry":
        rBndryPts = np.delete(rayDistances, noEntriesRadial) * parametersBC.scalingFactorK # scaling factor to match training environment 16/rayLen from pyBullet_environment
    else:
        rBndryPts = np.delete(rayDistances, noEntriesRadial) * 16

    egocentricBCActivity = BCActivity.boundaryCellActivitySimulation(thetaBndryPts, rBndryPts)
    ratesHDC_Amir = np.array(rates_hdc)
    rescaler = max(rates_hdc)
    # Not sure whether that clipping is biologically plausible but without it to many tr Layers are activated
    ratesHDC_Amir = np.where(ratesHDC_Amir / rescaler >= 0.8, ratesHDC_Amir / rescaler, 0)
    ratesHDCSimple = HDCActivity.headingCellsActivityTraining(decodedDir) # gaussian for each neurons activity
    transLayers, bvcActivity = BCsimulation.calculateActivities(egocentricBCActivity, ratesHDCSimple)

    afterBCStep = time.time()
    bcTimes.append(afterBCStep-beforeBCStep)

    ## for difference and plotting values
    eBCSummed += np.sum(egocentricBCActivity)
    bvcSummed += np.sum(bvcActivity)
    diff = np.sum(bvcActivity) - np.sum(egocentricBCActivity)
    eachRateDiff.append(diff)

    # current position
    position = env.getPosition()

    xPositions.append(position[0])
    yPositions.append(position[1])
    eBCRates.append(np.sum(egocentricBCActivity))
    bvcRates.append(np.sum(bvcActivity))
    if t % 20 == 0:
        sampleX.append(position[0])
        sampleY.append(position[1])
        sampleT.append(str(int(round(t))))

    ################################################    end BC-Model    ################################################

    # plotting
    if time.time() > nextplot and rtplot:
        nextplot += 1.0 / plotfps
        beforeStep = time.time()
        ###############     BC-Model    ##############
        if plotAllLayers:
            bcPlotter.bcPlotting(egocentricBCActivity, bvcActivity, transLayers[:, 0], transLayers[:, 1], transLayers[:, 2],
                                 transLayers[:, 3], transLayers[:, 4], transLayers[:, 5], transLayers[:, 6], transLayers[:, 7],
                                 transLayers[:, 8], transLayers[:, 9], transLayers[:, 10], transLayers[:, 11], transLayers[:, 12],
                                 transLayers[:, 13], transLayers[:, 14], transLayers[:, 15], transLayers[:, 16], transLayers[:, 17],
                                 transLayers[:, 18], transLayers[:, 19], decodedDir)
        else:
            bcPlotter.bcPlotting(egocentricBCActivity, bvcActivity, transLayers[:, 5], transLayers[:, 15], transLayers[:, 14], decodedDir)
        ################    end BC-Model    ##########

        # This plots the decoded heading, simulation speed will decrease when plotting
        #plotter.plot(rates_hdc, rates_sl, rates_sr, stimL, stimR, realDir, decodedDir)
        afterStep = time.time()
        plotTimes.append((afterStep - beforeStep))
    afterStep = time.time()
    t_ctr += 1

# final calculations
t_total = time.time() - t_before
X = np.arange(0.0, t_episode, dt_robot)
cahv = [avs[i] - avs[i - 1] if i > 0 else avs[0] for i in range(len(avs))]
cerr = [errs_signed[i] - errs_signed[i - 1] if i > 0 else errs_signed[0] for i in range(len(errs_signed))]
corr, _ = pearsonr(cahv, cerr)

#BC-Model
meanDiff = (bvcSummed - eBCSummed) / (t_episode / dt_robot)
maxDiff = max(np.subtract(np.array(bvcRates), np.array(eBCRates)))

# error noisy integration vs. noisy HDC
if use_noisy_av:
    plt.plot(X, errs_noisy_signed, label="Noisy integration")
    plt.plot(X, errs_signed, label="Noisy HDC")
    plt.plot([0.0, t_episode], [0.0, 0.0], linestyle="dotted", color="k")
    plt.xlabel("time (s)")
    plt.ylabel("error (deg)")
    plt.legend()
    plt.show()

# print results
print("\n\n\n")
print("############### Begin Simulation results ###############")
#BC-Model
print("Mean difference between eBC and BVC firing: " + str(meanDiff))
print("Max difference in firing: " + str(maxDiff))
print("max eBC activity: " + str(max(eBCRates)))
print("max BVC activity: " + str(max(bvcRates)))

# performance tracking
print("Total time (real): {:.2f} s, Total time (simulated): {:.2f} s, simulation speed: {:.2f}*RT".format(t_total, t_episode, t_episode / t_total))
print("Average step time network:  {:.4f} ms; {} it/s possible".format(1000.0 * np.mean(netTimes), 1.0/np.mean(netTimes)))
print("Average step time robot:    {:.4f} ms; {} it/s possible".format(1000.0 * np.mean(robotTimes), 1.0/np.mean(robotTimes)))
if rtplot:
    print("Average step time plotting: {:.4f} ms; {} it/s possible".format(1000.0 * np.mean(plotTimes), 1.0/np.mean(plotTimes)))
time_coverage = 0.0
print("Average time decoding:      {:.4f} ms; {} it/s possible".format(1000.0 * np.mean(decodeTimes), 1.0/np.mean(decodeTimes)))
print("Steps done network:  {}; Time: {:.3f} s; {:.2f}% of total time".format(len(X) * timesteps_neuron, len(X) * timesteps_neuron * np.mean(netTimes), 100 * len(X) * timesteps_neuron * np.mean(netTimes) / t_total))
time_coverage += 100 * len(X) * timesteps_neuron * np.mean(netTimes) / t_total
print("Steps done robot:    {}; Time: {:.3f} s; {:.2f}% of total time".format(len(X), len(X) * np.mean(robotTimes), 100 * len(X) * np.mean(robotTimes) / t_total))
time_coverage += 100 * len(X) * np.mean(robotTimes) / t_total
if rtplot:
    print("Steps done plotting: {}; Time: {:.3f} s; {:.2f}% of total time".format(int(t_episode / plotfps), int(t_episode / plotfps) * np.mean(plotTimes), 100 * int(t_episode / plotfps) * np.mean(plotTimes) / t_total))
    time_coverage += 100 * int(t_episode / plotfps) * np.mean(plotTimes) / t_total
print("Steps done decoding: {}; Time: {:.3f} s; {:.2f}% of total time".format(len(X), len(X) * np.mean(decodeTimes), 100 * len(X) * np.mean(decodeTimes) / t_total))
time_coverage += 100 * len(X) * np.mean(decodeTimes) / t_total
print("Steps done bc simulation: {}; Time: {:.3f} s; {:.2f}% of total time".format(len(X), len(X) * np.mean(bcTimes), 100 * len(X) * np.mean(bcTimes) / t_total))
time_coverage += 100 * len(X) * np.mean(bcTimes) / t_total
print("Time covered by the listed operations: {:.3f}%".format(time_coverage))
print("maximum angular velocity: {:.4f} deg/s".format(max(avs) * r2d))
print("average angular velocity: {:.4f} deg/s".format(sum([r2d * (x / len(avs)) for x in avs])))
print("median angular velocity:  {:.4f} deg/s".format(np.median(avs)))
print("maximum error: {:.4f} deg".format(max(errs)))
print("average error: {:.4f} deg".format(np.mean(errs)))
print("median error:  {:.4f} deg".format(np.median(errs)))
print("################ End Simulation results ################")
print("\n\n\n")

# close real-time plot
plt.close()
plt.ioff()

# plot error and angular velocity
fig, ax1 = plt.subplots()
#commented out lines provide graphics for HDC network
''''
# ax1.set_xlim(200, 375)
ax1.set_xlabel("time (s)")
ax1.set_ylabel("error (deg)")
ax1.set_ylim(-13.5, 13.5)
ax1.plot(X, errs_signed, color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax2 = ax1.twinx()
ax2.set_ylabel("angular velocity (deg/s)")
ax2.set_ylim(-50, 50)
ax2.plot(X, [x * r2d for x in avs], color="tab:orange")
ax2.tick_params(axis="y", labelcolor="tab:orange")
ax1.plot([0.0, t_episode], [0.0, 0.0], linestyle="dotted", color="k")
fig.tight_layout()
'''

ax3 = plt.axes(projection='3d')
ax3.set_title("rate differences + trajectory 3d")
ax3.set_xlim(-5, 10)
ax3.set_ylim(-10, 5)
ax3.set_xlabel("x Position")
ax3.set_ylabel("y Position")
ax3.set_zlabel("rate differences")
ax3.plot(xPositions, yPositions, eachRateDiff)
plt.show()

# plot rate differences
plt.title("Rate Differences")
plt.xlabel("time (s)")
plt.ylabel("rate difference")
plt.xlim(0.0, t_episode)
plt.plot(X, eachRateDiff)
plt.savefig('rate_Differences_curved.pdf')
plt.show()

#plot BVC and eBC rates
plt.title("Total firing rates at each timestep")
plt.xlabel("time (s)")
plt.ylabel("firing rate")
plt.xlim(0.0, t_episode)
plt.plot(X, eBCRates, color="blue", label="eBC")
plt.plot(X, bvcRates, color="orange", label="TR layer 6")
plt.savefig('total_rates_curved.pdf')
plt.legend()
plt.show()

#plot trajectory with timestamps
plt.title("Trajectory")
plt.xlabel("x Position")
plt.ylabel("y Position")
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.plot(xPositions, yPositions)


font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 6 }
matplotlib.rc('font', **font)
plt.scatter(sampleX, sampleY, s=5)
plt.xlim(-10, 10)
plt.ylim(-10, 10)
for i in range(int(t_episode / 20)):
    str = sampleT[i] + "(s)"
    plt.text(sampleX[i], sampleY[i], str)
plt.savefig('trajectory_curved.pdf')
plt.show()
'''
# plot only error
plt.xlabel("time (s)")
plt.ylabel("error (deg)")
plt.ylim(-1.6, 1.6)
plt.xlim(0.0, t_episode)
plt.plot(X, errs_signed)
plt.plot([0.0, t_episode], [0.0, 0.0], linestyle="dotted", color="k")
plt.show()

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

# plot change in angular velocity vs. change in error
plt.scatter(cahv, cerr)
plt.plot([min(cahv), max(cahv)], [corr * min(cahv), corr * max(cahv)], label="linear approximation with slope {:.2f}".format(corr), color="tab:red")
plt.legend()
plt.xlabel("change in angular velocity (deg/s)")
plt.ylabel("change in error (deg)")
plt.show()
'''
if not run_from_data:
    env.close()