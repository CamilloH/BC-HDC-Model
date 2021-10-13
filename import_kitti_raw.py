import numpy as np
import matplotlib.pyplot as plt
from helper import angleDist
import pickle
from datetime import datetime
import re
from os import listdir

# basedir containing kitti sequence folders
basedir_in = "/home/amir/kitti/kitti_dataset_raw"

def parse(sequence):
    outfile = "data/avs_kitti_raw_{}.p".format(sequence)
    # parse times
    times_in = open("{}/{}/oxts/timestamps.txt".format(basedir_in, sequence), "r")
    instrs = times_in.read().split("\n")
    times_in.close()
    datetimes = []
    def splitStringMultipleSeperators(s, seperators):
        strs = [s]
        strs_new = []
        for sep in seperators:
            for st in strs:
                for x in st.split(sep):
                    strs_new.append(x)
            strs = strs_new
            strs_new = []
        return strs
    for s in instrs:
        if s != "":
            tokens = splitStringMultipleSeperators(s, [" ", ":", "-", "."])
            year = int(tokens[0])
            month = int(tokens[1])
            day = int(tokens[2])
            hour = int(tokens[3])
            minute = int(tokens[4])
            second = int(tokens[5])
            microsecond = int(int(tokens[6]) / 1000)
            datetimes.append(datetime(year, month, day, hour, minute, second, microsecond))
    # set initial time to 0.0
    times = [0.0]
    for d in datetimes[1:len(datetimes)]:
        times.append((d - datetimes[0]).total_seconds())

    # parse angular velocities and directions
    dataset_size = len(times)
    def readDataset(number):
        f_in = open("{}/{}/oxts/data/{:010d}.txt".format(basedir_in, sequence, number), "r")
        in_text = f_in.read()
        in_tokens = in_text.split()
        f_in.close()
        return {"direction" : float(in_tokens[5]), "av" : float(in_tokens[22])}

    # set initial direction to 0.0
    directions = []
    initialDirection = readDataset(0)["direction"]
    avs = []
    for i in range(dataset_size):
        ds = readDataset(i)
        directions.append((ds["direction"] - initialDirection) % (2*np.pi))
        avs.append(ds["av"])

    out = (times, avs, directions)
    with open(outfile, 'wb') as fl:
        fl.write(pickle.dumps(out))
        fl.close()
    print("Dataset {} imported. Time: {:.2f} s".format(sequence, times[-1]))
    print("Output file: {}".format(outfile))

sequences = listdir(basedir_in)
for i in range(len(sequences)):
    print("Importing dataset {}/{}...".format(i + 1, len(sequences)))
    parse(sequences[i])
