import rospy
import scipy.spatial.transform.rotation as R
from std_msgs.msg import String
import numpy as np
import helper
from parametersHDC import n_hdc, weight_av_stim
from hdc_template import generateHDC

from sensor_msgs.msg import _Imu

#####################################################################
# This file is for directly processing ROS IMU messages.            #
# It provides no output except console.                             #
# It's only used to verify real-time performance on a raspberry pi. #
#####################################################################

# rad to deg
r2d = 180.0 / np.pi

# output arrays
timestamps = []
orientations = []
avs = []

class SimState:
    def __init__(self):
        # for checking elapsed time
        self.lastTimestamp = 0.0

        # the first callback initializes these
        self.initialization_done = False
        self.initial_orientation = 0.0
        self.initial_timestamp = 0.0

        ### result arrays ###
        # hdc network
        self.errs_signed = []
        self.errs = []
        self.decDirections = []

        # numerical quadrature of angular velocity
        # trapezoid rule is used
        # quad_thetas: changes in angle per timestep
        self.quad_thetas = []
        self.quad_errs_signed = []
        self.quad_errs = []
        self.quad_dirs = []
        self.quad_dir = 0.0
        #####################


        # init HDC network at 0.0
        self.hdc = generateHDC()

simState = SimState()
print("Initialization done")

def callback(data_in):
    global simState
    # get orientation
    ori = data_in.orientation
    ori_quat = np.array((ori.x, ori.y, ori.z, ori.w))
    ori_rot = R.Rotation.from_quat(ori_quat)
    orientation = ori_rot.as_euler("zyx", degrees=False)[0]

    # get angular velocity
    av = data_in.angular_velocity.z

    # get timestamp
    timestamp = data_in.header.stamp.secs + data_in.header.stamp.nsecs / 1E9

    # do initialization if not already done
    if not simState.initialization_done:
        simState.initialization_done = True
        simState.initial_orientation = orientation
        simState.initial_timestamp = timestamp
        simState.lastTimestamp = timestamp
        # do nothing for the first timestep
        return
    
    dt = timestamp - simState.lastTimestamp

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
    stimL = getStimL(av)
    stimR = getStimR(av)
    simState.hdc.setStimulus('hdc_shift_left', lambda _ : stimL)
    simState.hdc.setStimulus('hdc_shift_right', lambda _ : stimR)

    # HDC network step
    simState.hdc.step(dt)

    # get rates
    rates_hdc = simState.hdc.getLayer('hdc_attractor')

    # decode direction and calculate error
    direc = (orientation - simState.initial_orientation) % (2*np.pi)
    decodedDir = helper.decodeAttractorNumpy(rates_hdc)
    simState.decDirections.append(decodedDir)
    simState.err_signed_rad = helper.angleDist(direc, decodedDir)
    simState.errs_signed.append(r2d * simState.err_signed_rad)
    simState.errs.append(abs(r2d * simState.err_signed_rad))
    print("Time: {:.2f} s, Error: {:.2f} deg".format(timestamp - simState.initial_timestamp, r2d * simState.err_signed_rad))

    # set last timestamp
    simState.lastTimestamp = timestamp

rospy.init_node("python_listener", anonymous=True)
rospy.Subscriber("/imu_bosch/data", _Imu.Imu, callback)
print("Connected to ROS!")
while not rospy.core.is_shutdown():
    rospy.rostime.wallsleep(0.5)
print("Exited ROS")