import BCActivity
import pybullet as p
import gym
import signal
import sys
import pybullet_data
import math
import numpy as np
import parametersBC
from scipy.spatial.transform import Rotation as R
import time

def collision():
    p.btConvexShape

def signal_handler(signal, frame):
    print("\nprogram exiting gracefully")
    sys.exit(0)

def angleDist(a, b):
    reala = a % (2 * np.pi)
    realb = b % (2 * np.pi)
    return min(abs(reala - realb), abs(reala - (2*np.pi - realb)))

class PybulletEnvironment(gym.Env):
    def __init__(self, rate, visualize, model):
        self.model = model
        self.visualize = visualize
        self.rate_ = rate
        # connect to pybullet
        if self.visualize:
            p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            p.connect(p.DIRECT)
        p.setTimeStep(1.0/self.rate_)  # default is 240 Hz
        # reset camera angle
        if self.model == "maze" or "curved" or "eastEntry":
            p.resetDebugVisualizerCamera(cameraDistance=4.5, cameraYaw=0, cameraPitch=-70, cameraTargetPosition=[0.55, -7.35, 5.0]) # (cameraDistance=4.5, cameraYaw=0, cameraPitch=-70, cameraTargetPosition=[0.55, -7.35, 5.0]) for p3dx shot (cameraDistance=4.5, cameraYaw=90, cameraPitch=-30, cameraTargetPosition=[-20, -20, 2.0])
        else:
            p.resetDebugVisualizerCamera(cameraDistance=4.5, cameraYaw=0, cameraPitch=-70, cameraTargetPosition=[8, -4, 0.2])
        self.carId = []
        self.planeId = []
        self.action_space = []
        self.observation_space = []
        self.euler_angle = 0
        self.euler_angle_before = 0
        # addition for BC-Model helper that returns coordinates of encountered boundary segments wth getRays()
        self.raysThatHit = []
        pass

    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1, p.COV_ENABLE_GUI, 0)
        self.euler_angle_before = self.euler_angle
        self.braitenberg()
        # step simulation
        p.stepSimulation()


        # return change in orientation
        # before
        e_b = self.euler_angle_before[2]
        # after
        e_a = self.euler_angle[2]
        # fix transitions pi <=> -pi
        # in top left quadrant
        e_b_topleft = e_b < np.pi and e_b > np.pi / 2
        e_a_topleft = e_a < np.pi and e_a > np.pi / 2
        # in bottom left quadrant
        e_b_bottomleft = e_b < -np.pi / 2 and e_b > -np.pi
        e_a_bottomleft = e_a < -np.pi / 2 and e_a > -np.pi
        if e_a_topleft and e_b_bottomleft:
            # transition in negative direction
            return -(abs(e_a - np.pi) + abs(e_b + np.pi))
        elif e_a_bottomleft and e_b_topleft:
            # transition in positive direction
            return abs(e_a + np.pi) + abs(e_b - np.pi)
        else:
            # no transition, just the difference
            return e_a - e_b

    def reset(self):
        # reset simulation
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # we will enable rendering after we loaded everything
        p.setGravity(0, 0, -10 * 1)

        # reload model
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        if self.model == "maze":
            self.planeId = p.loadURDF("maze_2_2_lane/plane.urdf")
            self.carId = p.loadURDF("p3dx/urdf/pioneer3dx.urdf", basePosition=[7.7, -10, 0.02], baseOrientation=[0.0, 0.0, 0.7071067811865475, 0.7071067811865475])
            cubeStartOrientation = p.getQuaternionFromEuler([0, 0, np.pi/2])
        elif self.model == "plus":
            self.planeId = p.loadURDF("p3dx/plane/plane.urdf", globalScaling=2.5)
            self.carId = p.loadURDF("p3dx/urdf/pioneer3dx.urdf", basePosition=[0, -5, 0.02])
        elif self.model == "curved":
            self.planeId = p.loadURDF("maze_curved_elements/plane.urdf")
            self.carId = p.loadURDF("p3dx/urdf/pioneer3dx.urdf", basePosition=[7.7, -10, 0.02], baseOrientation=[0.0, 0.0, 0.7071067811865475, 0.7071067811865475])
        else:
            self.planeId = p.loadURDF("maze_bottom_left_entry/plane.urdf")
            self.carId = p.loadURDF("p3dx/urdf/pioneer3dx.urdf", basePosition=[7.7, -10, 0.02], baseOrientation=[0.0, 0.0, 0.7071067811865475, 0.7071067811865475]) # -20, -24 for p3dx shot
        self.euler_angle = p.getEulerFromQuaternion(p.getLinkState(self.carId, 0)[1])
        print("Euler_angle_before: ", self.euler_angle_before)

        # render back
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)  # rendering's back on again
        observation = []
        return observation

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()

    def getPosition(self):
        position = p.getLinkState(self.carId, 0)[0]
        return position

    def braitenberg(self):
        detect = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        if self.model == "maze" or "curved":
            braitenbergL = np.array(
                [-0.8, -0.75, -0.7, -0.65, -0.6, -0.55, -0.5, -0.45, -0.4, -0.35, -0.3, -0.25, -0.2,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0,
                 -1.6, -1.55, -1.5, -1.45, -1.4, -1.35, -1.3, -1.25, -1.2, -1.15, -1.1, -1.05, -1.0])
            braitenbergR = np.array(
                [-1.0, -1.05, -1.1, -1.15, -1.2, -1.25, -1.3, -1.35, -1.4, -1.45, -1.5, -1.55, -1.6,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0,
                 -0.2, -0.25, -0.3, -0.35, -0.4, -0.45, -0.5, -0.55, -0.6, -0.65, -0.7, -0.75, -0.8])
            noDetectionDist = 1.75
            velocity_0 = 5.5
            maxDetectionDist = 0.25
        else:
            braitenbergL = np.array(
                [-0.8, -0.75, -0.7, -0.65, -0.6, -0.55, -0.5, -0.45, -0.4, -0.35, -0.3, -0.25, -0.2,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0,
                 -1.6, -1.55, -1.5, -1.45, -1.4, -1.35, -1.3, -1.25, -1.2, -1.15, -1.1, -1.05, -1.0])
            braitenbergR = np.array(
                [-1.0, -1.05, -1.1, -1.15, -1.2, -1.25, -1.3, -1.35, -1.4, -1.45, -1.5, -1.55, -1.6,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0,
                 -0.2, -0.25, -0.3, -0.35, -0.4, -0.45, -0.5, -0.55, -0.6, -0.65, -0.7, -0.75, -0.8])
            noDetectionDist = 1.0
            velocity_0 = 2.0
            maxDetectionDist = 0.2

        rayDist = self.ray_detection()
        for i in range(len(rayDist)):
            if 0 < rayDist[i] < noDetectionDist:
                # something is detected
                if rayDist[i] < maxDetectionDist:
                    rayDist[i] = maxDetectionDist
                # dangerous level, the higher, the closer
                detect[i] = 1.0 - 1.0 * ((rayDist[i] - maxDetectionDist) * 1.0 / (noDetectionDist - maxDetectionDist))
            else:
                # nothing is detected
                detect[i] = 0

        vLeft = velocity_0
        vRight = velocity_0

        # print(detect)
        for i in range(len(rayDist)):
            vLeft = vLeft + braitenbergL[i] * detect[i] * 1
            vRight = vRight + braitenbergR[i] * detect[i] * 1

        '''
        minVelocity = 0.5
        if abs(vLeft) < minVelocity and abs(vRight) < minVelocity:
            vLeft = minVelocity
            vRight = minVelocity
        print("V Left:", vLeft, "V Right", vRight)
        '''
        p.setJointMotorControlArray(bodyUniqueId=self.carId,
                                    jointIndices=[4, 6],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocities=[vLeft, vRight],
                                    forces=[10, 10])

    def ray_detection(self):
        # the index of the ray is from the front, counter-clock-wise direction #
        # detect range rayLen = 1 #
        p.removeAllUserDebugItems()
        rayReturn = []
        rayFrom = []
        rayTo = []
        rayIds = []
        numRays = 51
        if self.model=="maze" or "curved":
            # set rayLength in parametersBC
            rayLen = parametersBC.rayLength
        else:
            rayLen = 1
        rayHitColor = [1, 0, 0]
        rayMissColor = [1, 1, 1]

        replaceLines = True

        for i in range(numRays):
            # rayFromPoint = p.getBasePositionAndOrientation(self.carId)[0]
            rayFromPoint = p.getLinkState(self.carId, 0)[0]
            rayReference = p.getLinkState(self.carId, 0)[1]
            euler_angle = p.getEulerFromQuaternion(rayReference)  # in degree
            # print("Euler Angle: ", rayFromPoint)
            rayFromPoint = list(rayFromPoint)
            rayFromPoint[2] = rayFromPoint[2] + 0.02
            rayFrom.append(rayFromPoint)
            rayTo.append([
                rayLen * math.cos(
                    2.0 * math.pi * float(i) / numRays + 360.0 / numRays / 2 * math.pi / 180 + euler_angle[2]) +
                rayFromPoint[0],
                rayLen * math.sin(
                    2.0 * math.pi * float(i) / numRays + 360.0 / numRays / 2 * math.pi / 180 + euler_angle[2]) +
                rayFromPoint[1],
                rayFromPoint[2]
            ])

            # if (replaceLines):
            #     if i == 0:
            #         # rayIds.append(p.addUserDebugLine(rayFrom[i], rayTo[i], [0, 0, 1]))
            #         pass
            #     else:
            #         # rayIds.append(p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor))
            #         pass
            # else:
            #     rayIds.append(-1)

        results = p.rayTestBatch(rayFrom, rayTo, numThreads=0)
        for i in range(numRays):
            hitObjectUid = results[i][0]

            if (hitObjectUid < 0):
                hitPosition = [0, 0, 0]
                # p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor, replaceItemUniqueId=rayIds[i])
                if(i ==0):
                    p.addUserDebugLine(rayFrom[i], rayTo[i], (0,0,0))
                p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor)
                rayReturn.append(-1)
            else:
                hitPosition = results[i][3]
                # p.addUserDebugLine(rayFrom[i], hitPosition, rayHitColor, replaceItemUniqueId=rayIds[i])
                p.addUserDebugLine(rayFrom[i], rayTo[i], rayHitColor)
                rayReturn.append(
                    math.sqrt((hitPosition[0] - rayFrom[i][0]) ** 2 + (hitPosition[1] - rayFrom[i][1]) ** 2))

        self.euler_angle = euler_angle
        # print("euler_angle: ", euler_angle[2] * 180 / np.pi)

        ### BC-Model
        # returns the distance to the walls hit starting from 0 to 2pi counter clockwise so each of the 51 entries is
        # the length for one radial separation bin
        self.raysThatHit = rayReturn
        ###
        return rayReturn

    def getRays(self):
        return self.raysThatHit

    def euler_calculation(self):
        position, orientation = p.getBasePositionAndOrientation(self.carId)
        r = R.from_quat(list(orientation))
        euler_angle = r.as_euler('zyx', degrees=True)
        print("orientation: ", euler_angle)
        pass
