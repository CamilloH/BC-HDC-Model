import rospy
import scipy.spatial.transform.rotation as R
from std_msgs.msg import String
import numpy as np
import pickle
from tqdm import tqdm

from sensor_msgs.msg import _Imu, CompressedImage


# rad to deg
r2d = 180.0 / np.pi

# output arrays
timestamps = []
orientations = []
avs = []

timestamps_img = []
jpg_data_img = []


def callback(data_in):
    ori = data_in.orientation
    ori_quat = np.array((ori.x, ori.y, ori.z, ori.w))
    ori_rot = R.Rotation.from_quat(ori_quat)
    ori_euler = ori_rot.as_euler("zyx", degrees=False)
    orientations.append(ori_euler[0])

    av = data_in.angular_velocity
    avs.append(av.z)

    timestamp = data_in.header.stamp.secs + data_in.header.stamp.nsecs / 1E9
    timestamps.append(timestamp)
    print("IMU timestamp: {}".format(timestamp))

def callback_image(data_in):
    timestamp = data_in.header.stamp.secs + data_in.header.stamp.nsecs / 1E9
    '''
    with open('picture_out_preview.jpg', 'wb') as f:
        f.write(data_in.data)
        f.close()
    '''
    jpg_data_img.append(data_in.data)
    timestamps_img.append(timestamp)
    print("Cam timestamp: {}".format(timestamp))

rospy.init_node("python_listener", anonymous=True)
rospy.Subscriber("/imu_bosch/data", _Imu.Imu, callback)
rospy.Subscriber("raspicam_node/image/compressed", CompressedImage, callback_image)
while not rospy.core.is_shutdown():
    rospy.rostime.wallsleep(0.5)
print("Exited ROS")

# let time and orientation start at 0.0
times = [0.0]
for i in tqdm(range(1, len(timestamps))):
    times.append(timestamps[i] - timestamps[0])

times_img = [0.0]
for i in tqdm(range(1, len(timestamps_img))):
    times_img.append(timestamps_img[i] - timestamps_img[0])

directions = [0.0]
for i in tqdm(range(1, len(orientations))):
    directions.append((orientations[i] - orientations[0]) % (2*np.pi))

# pickle
# SET OUTFILE MANUALLY
# outfile = "data/avs_info_building.p"
outfile = "data/info_2020-11-07-20-44-57.p"
out = (times, avs, directions, times_img, jpg_data_img)
with open(outfile, 'wb') as fl:
    fl.write(pickle.dumps(out))
    fl.close()