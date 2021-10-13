from pybullet_environment import PybulletEnvironment
from tqdm import tqdm
import numpy as np
import pickle

# output file
outfile = "data/thetas_pybullet_maze_240Hz.p"
# total episode time in seconds
t_episode = 375
# robot timestep
dt_robot = 1.0/240.0
# simulation environment, available models: "maze", "plus"
env_model = "maze"
# the simulation environment window can be turned off, speeds up the simulation significantly
env_visualize = False

env = PybulletEnvironment(1/dt_robot, env_visualize, env_model)
env.reset()

thetas = []
for t in tqdm(np.arange(0.0, t_episode, dt_robot)):
    theta = env.step([])
    thetas.append(theta)

out = (thetas, t_episode, dt_robot)
with open(outfile, 'wb') as fl:
    fl.write(pickle.dumps(out))
    fl.close()