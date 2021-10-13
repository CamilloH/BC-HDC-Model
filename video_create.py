import time
import numpy as np
import math
from tqdm import tqdm
import hdcNetwork
from hdcAttractorConnectivity import HDCAttractorConnectivity
from network import NetworkTopology
import helper
from parametersHDC import n_hdc, weight_av_stim
from hdc_template import generateHDC
from scipy.stats import pearsonr
import scipy
import random
import pickle
import sys
import os
import re
from datetime import datetime
from SimResult import SimResult
from neuron import r_m
import matplotlib as mpl
mpl.use('AGG')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as image
import PIL.Image as PILImage
import io
import multiprocessing as mp


# ffmpeg is required to run this file.

###### INPUTS #####
# works with the pickle file created by video_import_ros.py
infile = 'data/info_2020-11-07-20-44-57.p'
# TRAJECTORY VIDEO: video_frames/input/trajectory.mp4
###################

# output: video_frames/output.mp4
# may take some time, every frame is plotted individually using matplotlib

#####  PARAMS  #####
trajectory_cropping_topleft = [110, 50]
trajectory_cropping_bottomright = [1750, 950]
start_time = 0.0
end_time = 1045.0
# for syncing the trajectory video
# offset given in frames, 24 FPS
trajectoryOffset = 0
# offset in degrees to align hdc view with trajectory
angleOffset = 113
num_threads = 12
# cap simulation time for long IMU delays
dt_cap = 0.1
####################



# all individual trajectory frames: video_frames/input/frames/%06d.jpg, 24 FPS
# get number of trajectory frames
num_trajectory_frames = len(os.listdir('video_frames/input/frames/'))
# extract if necessary
if num_trajectory_frames == 0:
    print("\nExtracting trajectory video...\n")
    os.system("ffmpeg -i video_frames/input/trajectory.mp4 -r 25 video_frames/input/frames/out-%06d.png")
    num_trajectory_frames = len(os.listdir('video_frames/input/frames/'))
print("\nTrajectory video extracted.\n")

def runSimulation(times, avs, directions, dt_neuron_min, label=""):
    r2d = 180.0 / np.pi
    t_episode = times[-1]

    # performance tracking
    netTimes = []
    stepCounterNet = 0
    decodeTimes = []
    netTimes = []
    stepCounterDecode = 0

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

    # for live plotting
    rate_lists_hdc = []
    rate_lists_sl = []
    rate_lists_sr = []
    #####################


    # init HDC network at 0.0
    hdc = generateHDC()

    # offset all directions by starting direction
    dir_offset = directions[0]

    for i in tqdm(range(1, len(times))):
        t = times[i]
        dt = times[i] - times[i - 1]
        # cap dt
        if dt > dt_cap:
            dt = dt_cap
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
        '''
        if dt > 0.02:
            print("high dt at t={}:{}; dt={}, av={}".format(int(t // 60), int(t % 60), dt, av))
            os.system("echo 'high dt at t={}:{}; dt={}, av={}' >> dts.txt".format(int(t // 60), int(t % 60), dt, av))
        '''
        beforeStep = time.time()
        stepCounterNet += timesteps_neuron
        # print(dt, dt_neuron, timesteps_neuron)
        hdc.step(dt_neuron, numsteps=timesteps_neuron)
        afterStep = time.time()
        netTimes.append((afterStep - beforeStep) / timesteps_neuron)

        # get rates
        rates_hdc = list(hdc.getLayer('hdc_attractor'))
        rates_sl = list(hdc.getLayer('hdc_shift_left'))
        rates_sr = list(hdc.getLayer('hdc_shift_right'))
        rate_lists_hdc.append(rates_hdc)
        rate_lists_sl.append(rates_sl)
        rate_lists_sr.append(rates_sr)

        # decode direction
        beforeStep = time.time()
        stepCounterDecode += 1
        decodedDir = helper.decodeAttractorNumpy(rates_hdc)
        decDirections.append(decodedDir)
        err_signed_rad = helper.angleDist(direc, decodedDir)
        errs_signed.append(r2d * err_signed_rad)
        errs.append(abs(r2d * err_signed_rad))
        afterStep = time.time()
        decodeTimes.append(afterStep - beforeStep)

    X = times[1:len(times)]
    result = SimResult(label, X, [x * r2d for x in avs], directions, decDirections, errs_signed, errs, quad_thetas, quad_dirs, quad_errs_signed, quad_errs)
    result.rate_lists_hdc = rate_lists_hdc
    result.rate_lists_sl = rate_lists_sl
    result.rate_lists_sr = rate_lists_sr
    return result

def loadImg(jpg_data):
    img = PILImage.open(io.BytesIO(jpg_data))
    return img.transpose(PILImage.ROTATE_90).transpose(PILImage.FLIP_LEFT_RIGHT)
    # img = scipy.ndimage.rotate(img, 90)

def loadTrajectoryImg(framenum):
    global num_trajectory_frames
    if framenum >= num_trajectory_frames:
        img = image.imread('video_frames/input/frames/out-{:06d}.png'.format(num_trajectory_frames - 1))
    else:
        img = image.imread('video_frames/input/frames/out-{:06d}.png'.format(framenum))
    return img[trajectory_cropping_topleft[1] : trajectory_cropping_bottomright[1], trajectory_cropping_topleft[0] : trajectory_cropping_bottomright[0]]

def plotFrame(args):
    (times, rate_list_hdc, rate_list_sl, rate_list_sr, decDirections_0centered, decDirections, realDirections_0centered, realDirections, avs, frameNumber, image_jpg, maxAbsErr, maxAbsAv, frameFileNumber, trajectoryFrameNumber) = args
    # fig = plt.figure(constrained_layout=True)
    fig = plt.figure()
    grid = fig.add_gridspec(9, 4)
    # trajectory
    ax0 = fig.add_subplot(grid[4:9, 1:3])
    ax0.xaxis.set_visible(False)
    ax0.yaxis.set_visible(False)
    img_trajectory = loadTrajectoryImg(trajectoryFrameNumber)
    ax0.imshow(img_trajectory)

    # orientation
    ax5 = fig.add_subplot(grid[0:2, :])
    ax5.set_ylabel("orientation\n(deg)")
    ax5.xaxis.set_visible(False)
    ax5.set_xlim([0.0, end_time])
    ax5.set_ylim([0, 360])
    ax5.plot(times[0 : frameNumber], decDirections[0 : frameNumber], label = "HDC")
    ax5.plot(times[0 : frameNumber], realDirections[0 : frameNumber], label = "IMU")
    ax5.legend()

    # angular velocity
    ax1 = fig.add_subplot(grid[2, :])
    ax1.set_ylabel("angular velocity\n(deg/s)")
    ax1.set_xlim([0.0, end_time])
    ax1.set_ylim([-maxAbsAv, maxAbsAv])
    ax1.xaxis.set_visible(False)
    ax1.plot(times[0 : frameNumber], avs[0 : frameNumber], color="g")

    # error
    ax2 = fig.add_subplot(grid[3, :])
    ax2.set_ylabel("error\n(deg)")
    ax2.set_xlabel("time (s)")
    ax2.set_xlim([0.0, end_time])
    ax2.set_ylim([-maxAbsErr, maxAbsErr])
    ax2.plot(times[0 : frameNumber], errors[0 : frameNumber], color="r")

    # camera
    ax3 = fig.add_subplot(grid[4:9, 0])
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])
    ax3.xaxis.set_visible(False)
    ax3.yaxis.set_visible(False)
    img_cam = loadImg(image_jpg)
    ax3.imshow(img_cam)

    # HDC view
    ax4 = fig.add_subplot(grid[4:9, 3], projection="polar")
    invert = False
    # bias = -0.5*np.pi
    bias = angleOffset * (np.pi / 180)
    live_plot_X = [(x + bias) * (-1 if invert else 1) % (np.pi * 2) for x in np.linspace(0.0, np.pi * 2, n_hdc + 1)]
    ax4.plot(live_plot_X, [*rate_list_hdc, rate_list_hdc[0]], 'r-', label="HDC layer")
    ax4.plot(live_plot_X, [*rate_list_sl, rate_list_sl[0]], 'b-', label="shift-left layer")
    ax4.plot(live_plot_X, [*rate_list_sr, rate_list_sr[0]], 'g-', label="shift-right layer")
    # compass True orientation
    ax4.plot([realDirections[frameNumber] * (np.pi / 180), realDirections[frameNumber] * (np.pi / 180)], [0.0, r_m], 'k-', label="orientation (IMU)")
    # compass Decoded orientation
    ax4.plot([decDirections[frameNumber] * (np.pi / 180), decDirections[frameNumber] * (np.pi / 180)], [0.0, r_m], 'm-', label="orientation (HDC)")
    ax4.legend()

    # plt.tight_layout()
    fig.set_size_inches(16, 12)
    plt.subplots_adjust(left=0.07, right=0.97, top=0.99, bottom=0.01)
    plt.savefig("video_frames/frame_{:06d}".format(frameFileNumber))
    plt.close()
    # img_cam.close()
    # img_trajectory.close()
    return frameNumber

# flip directions and convert to degrees, also thin out resolution to 5%, there aren't enough pixels for 100k data points
# called once with SimResult to preprocess everything
def preprocessData(simResult):
    # thin out
    resolution = 20
    real_directions = []
    times = []
    dec_directions = []
    errs_signed = []
    avs_ = []
    rate_lists_hdc = []
    rate_lists_sl = []
    rate_lists_sr = []
    for i in range(len(simResult.times)):
        if i % resolution == 0:
            times.append(simResult.times[i])
            real_directions.append((simResult.realDirections[i] + angleOffset * (np.pi / 180)) % (2 * np.pi))
            dec_directions.append((simResult.decDirections[i] + angleOffset * (np.pi / 180)) % (2 * np.pi))
            errs_signed.append(simResult.errs_signed[i])
            avs_.append(simResult.avs[i])
            rate_lists_hdc.append(simResult.rate_lists_hdc[i])
            rate_lists_sl.append(simResult.rate_lists_sl[i])
            rate_lists_sr.append(simResult.rate_lists_sr[i])
    # preprocess
    realDirections_0centered = np.array([-x * (180.0 / np.pi) if x < np.pi else -(x - 2 * np.pi) * (180.0 / np.pi) for x in real_directions])
    decDirections_0centered = np.array([-x * (180.0 / np.pi) if x < np.pi else -(x - 2 * np.pi) * (180.0 / np.pi) for x in dec_directions])
    realDirections = np.array(real_directions) * (180 / np.pi)
    decDirections = np.array(dec_directions) * (180 / np.pi)
    errors = np.array(errs_signed)
    avs = np.array(avs_)
    return (times, realDirections, realDirections_0centered, decDirections, decDirections_0centered, errors, avs, rate_lists_hdc, rate_lists_sl, rate_lists_sr)

# load pickle file
print("loading input...")
f_in = open(infile, "rb")
(times, avs, directions, times_img, jpg_data_img) = pickle.load(f_in, encoding="bytes")
f_in.close()

# run full simulation first
print("running simulation...")
simResult = runSimulation(times, avs, directions, 0.0005)

# pre-process data
(times, realDirections, realDirections_0centered, decDirections, decDirections_0centered, errors, avs, rate_lists_hdc, rate_lists_sl, rate_lists_sr) = preprocessData(simResult)
t_b = 0.0
for t in times:
    if t < t_b:
        t_b = t
        print("TIMES NOT SORTED")
t_b = 0.0
for t in times_img:
    if t < t_b:
        t_b = t
        print("IMAGE TIMES NOT SORTED")

# plot frame-by-frame
print("plotting frames ...")
video_fps = 25
if end_time == None:
    end_time = min([times[-1], (num_trajectory_frames + trajectoryOffset) / 24])
maxAbsErr = max([abs(min(errors)), abs(max(errors))])
maxAbsAv = max([abs(min(avs)), abs(max(avs))])
# create video every 1000 frames
framesInVideo = 1000
videoCnt = 0
plottingArgs = []

frameNumber = 0
currentCamFrame = 0
for frameCnt, t in enumerate(tqdm(np.arange(start_time, end_time, 1.0 / video_fps))):
    # track frame number and cam frame
    if frameNumber < len(times) - 1 and t > times[frameNumber + 1]:
        frameNumber += 1
    if currentCamFrame < len(times_img) - 1 and t > times_img[currentCamFrame + 1]:
        currentCamFrame += 1
    # add args to list
    image_jpg = jpg_data_img[currentCamFrame]
    rate_list_hdc = rate_lists_hdc[frameNumber]
    rate_list_sl = rate_lists_sl[frameNumber]
    rate_list_sr = rate_lists_sr[frameNumber]
    plottingArgs.append((times, rate_list_hdc, rate_list_sl, rate_list_sr, decDirections_0centered, decDirections, realDirections_0centered, realDirections, avs, frameNumber, image_jpg, maxAbsErr, maxAbsAv, frameCnt % framesInVideo, max([1, trajectoryOffset + frameCnt])))
    if frameCnt % framesInVideo == framesInVideo - 1:
        # start thread pool
        p = mp.Pool(num_threads)
        f = p.map_async(plotFrame, plottingArgs)
        f.wait()
        # close pool to resolve memory issues
        p.close()
        # use ffmpeg to glue all frames together into an mp4
        ffmpeg_call = "ffmpeg -i video_frames/frame_{}06d.png video_frames/video_{:06d}.mp4".format(r"%", videoCnt)
        os.system(ffmpeg_call + " > ffmpeg_output.txt")
        # delete frames
        os.system("rm video_frames/*.png")
        videoCnt += 1
        plottingArgs = []

frameCnt = len(np.arange(start_time, end_time, 1.0 / video_fps)) - 1
# last batch
if frameCnt % framesInVideo != 0:
    # start thread pool
    p = mp.Pool(num_threads)
    f = p.map_async(lambda x : plotFrame(*x), plottingArgs)
    f.wait()
    # close pool to resolve memory issues
    p.close()
    # use ffmpeg to glue all frames together into an mp4
    ffmpeg_call = "ffmpeg -i video_frames/frame_{}06d.png video_frames/video_{:06d}.mp4".format(r"%", videoCnt)
    os.system(ffmpeg_call + " > ffmpeg_output.txt")
    # delete frames
    os.system("rm video_frames/*.png")

# use ffmpeg to glue all videos together
def concatVideos(video_dir):
    # get input videos
    video_names = os.listdir(video_dir)
    video_names.sort()
    video_names = list(filter(lambda x : re.match(r"video_[0-9]{6}.mp4$", x), video_names))
    # convert all to ts
    for name in video_names:
        os.system("ffmpeg -i {}/{} -c copy -bsf:v h264_mp4toannexb -f mpegts {}/{}_intermediate.ts".format(video_dir, name, video_dir, name))
    str_concatall = '"concat:' + "|".join(["{}/{}_intermediate.ts".format(video_dir, name) for name in video_names]) + '"'

    # concatenate
    os.system("ffmpeg -i {} -c copy -bsf:a aac_adtstoasc -y {}/output.mp4".format(str_concatall, video_dir))

    # cleanup
    os.system("rm {}/*.ts".format(video_dir))

print("creating video...")
concatVideos("video_frames")

# remove video clips
print("cleaning up...")
os.system("rm video_frames/video_*.mp4")