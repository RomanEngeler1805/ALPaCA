import pybullet as p
import gym
from gym import spaces
import numpy as np
import pybullet_data
import time
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt


class PushEnv(gym.Env):
    """
    Action: [0 -> stay, 1 -> left, 2 -> right, 3 -> forward, 4 -> backward]
    Observation: [x, y]
    Reward: 1/(target_distance+ 1) in [0, 1]
    """

    def __init__(self):
        
        self.rew_scale = 1.
        
        # domain boundaries
        self.minp = 0.0
        self.maxp = 0.4

        self.maxv = 1.5

        self.arm_length = 0.5 * self.maxp
        self.deacceleration = 0.4
        # target position
        self.target_position = np.r_[0.8 * self.maxp, 0.5 * self.maxp]

        self.low = np.r_[self.minp, self.minp, -self.maxv, -self.maxv, -0.2*self.maxp, -0.2*self.maxp, -self.maxv, -self.maxv]
        self.high = np.r_[self.maxp, self.maxp, +self.maxv, +self.maxv, 0.2*self.maxp, 0.2*self.maxp, +self.maxv, +self.maxv]

        # parameters for the simulation
        self.velocity_increment = 0.04
        self.control_hz = 30.
        self.sim_hz = 240.
        self.max_force = 400 # force of manipulator

        # define spaces in gym env
        self.action_space = spaces.Discrete(1)
        self.observation_range = spaces.Box(self.low, self.high)

        # initialize pybullet
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # action set
        self.velocities = np.asarray([
            [0.0, 0.0],
            [0.0, -1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.74, 0.74],
            [0.74, -0.74]
        ]) * self.velocity_increment

        self.reset()

    def reset(self):
        # reset simulation
        p.resetSimulation()
        p.setGravity(0.0, 0.0, -10.0)

        # load environment
        p.loadURDF("plane.urdf")

        # displacement
        displacement = (0.3+ np.random.rand()* 0.4) * self.maxp

        # load manipulation object
        self.object_id = p.loadURDF("urdfs/cuboid1.urdf", [0.05, displacement, 0.01])
        # load manipulator
        self.robot_position = np.r_[0.02, displacement+ 0.01*(-0.5 + np.random.rand()), 0.03] # XXXX
        rot = Rotation.from_rotvec(np.r_[0.0, 1.0, 0.0] * 0.5 * np.pi)
        self.robot_id = p.loadURDF("urdfs/cylinder.urdf", self.robot_position, rot.as_quat())

        # define constraint to track motion
        self.constraint_uid = p.createConstraint(self.robot_id, -1, -1, -1, p.JOINT_FIXED, [0.0, 0.0, 0.0],
                                                 [0., 0., 0.], self.robot_position, [0., 0., 0., 1.0], rot.as_quat())

        # object COM offset
        offset = 0.02*(-1. + 2.* np.random.rand())
        self.obj_offset_COM_local = np.array([0., offset])

        # reset velocity vector
        self.velocity_vector = np.zeros(2)

        # return observation
        obs = self._get_obs()

        return obs

    def step(self, action):
        # number of steps (control frequency < simulation frequency)
        N_steps = int(self.sim_hz / self.control_hz)

        # accelerate
        dvelocity = self.velocities[action] / N_steps

        # continuously track motion
        for _ in range(N_steps):
            self.velocity_vector += dvelocity
            self.robot_position[:2] += self.velocity_vector / self.sim_hz

            p.changeConstraint(self.constraint_uid, jointChildPivot=self.robot_position, maxForce=self.max_force)
            p.stepSimulation()
            #time.sleep(1./10.)

        # get observation
        obs = self._get_obs()

        done = False

        # check domain violations
        if np.any(obs < self.low) or np.any(obs > self.high):
            done = True

        # if robot moves too far away from object
        #if np.linalg.norm(obs[:2]- obs[4:6]) > 0.2* self.maxp:
        #    done = True

        # calculate reward
        reward = .01 / (.01 + np.linalg.norm(obs[:2]+ obs[4:6] - self.target_position)) # 1/ dist(object_to_target)
        #reward = self.rew_scale* (0.3- np.linalg.norm(obs[4:6] - self.target_position))

        return obs, reward, done, {}

    def _get_obs(self):
        # get object position
        pos_robot, _ = p.getBasePositionAndOrientation(self.robot_id)
        pos_object, rot_object_quat = p.getBasePositionAndOrientation(self.object_id)

        # add COM offset to object position
        r = Rotation.from_quat(rot_object_quat)
        rot_object_matrix = r.as_dcm()
        obj_offset_COM_global = np.dot(rot_object_matrix[:2,:2], self.obj_offset_COM_local)
        pos_object_COM = np.asarray(pos_object[:2])+ obj_offset_COM_global
        pos_object_COM -= pos_robot[:2]

        vel_object_lin, _ = p.getBaseVelocity(self.object_id)
        vel_robot_lin, _ = p.getBaseVelocity(self.robot_id)

        return np.concatenate([np.asarray(pos_robot[:2]),
                               self.velocity_vector,
                               pos_object_COM,
                               np.asarray(vel_robot_lin[:2])])

if __name__ == '__main__':
    # for some basic tests
    env = PushEnv()
    env.reset()
    done = False
    step = 0

    track_arm = []
    track_obj = []
    track_COM = []


    for i in range(30):
        # action = np.random.uniform(0.0, 1.0, (5,))
        action = [3 if step < 50 else 2][0]
        obs, reward, _, _ = env.step(action)
        obs, reward, _, _ = env.step(action)
        step += 1
        track_arm.append(obs[:2])
        track_obj.append(obs[4:6])
        track_COM.append(obs[6:])

    track_arm = np.asarray(track_arm)
    track_obj = np.asarray(track_obj)
    track_COM = np.asarray(track_COM)

    plt.figure()
    #plt.scatter(track_arm[:,0], track_arm[:,1])
    plt.scatter(track_obj[:, 0], track_obj[:, 1], color='b')
    plt.scatter(track_COM[:, 0], track_COM[:, 1], color='r')
    plt.show()
