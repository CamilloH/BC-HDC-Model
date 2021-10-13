import numpy as np
import matplotlib.pyplot as plt
from helper import angleDist
import pickle

basedir = '~/kitti/kitti_dataset'
sequence = '10'
outfile = 'data/thetas_kitti_{}.p'.format(sequence)
# parse times
times_in = open("{}/sequences/{}/times.txt".format(basedir, sequence), "r")
times = times_in.read().split()
dts = np.zeros(len(times) - 1, dtype=np.float32)
dt = 0.0
for i in range(1, len(times)):
    dts[i - 1] = float(times[i]) - float(times[i - 1])
    dt +=  dts[i - 1] * (1.0/(len(times) - 1))

f_in = open("{}/{}.txt".format(basedir, sequence), "r")

lst = f_in.read().split()
tVecs = np.zeros( (int(len(lst) / 12), 3) )
fVecs = np.zeros( (int(len(lst) / 12), 2) )
dirs = []
for i in range(12, len(lst), 12):
    tVecs[int(i / 12)][0] = lst[i + 3]
    tVecs[int(i / 12)][1] = lst[i + 7]
    tVecs[int(i / 12)][2] = lst[i + 11]
    fVecs[int(i / 12)][0] = lst[i + 2]
    fVecs[int(i / 12)][1] = lst[i + 10]
    dirs.append(np.arctan2(fVecs[int(i / 12)][0], fVecs[int(i / 12)][1]))

dataset_size = int(len(lst) / 12) - 1
thetas = []
avs = []
for i in range(1, dataset_size):
    thetas.append(angleDist(dirs[i - 1], dirs[i]))

print("Time: {}".format(dt * (dataset_size - 1)))
out = (thetas, dt * (dataset_size - 1) , dt)
with open(outfile, 'wb') as fl:
    fl.write(pickle.dumps(out))
    fl.close()