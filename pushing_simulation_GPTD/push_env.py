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
        ''''''
        # domain boundaries -------------------------------------
        # position
        self.minp = 0.0
        self.maxp = 0.6
        # velocity
        self.maxv = 1.5
        # robot arm length
        self.arm_length = 0.5 * self.maxp
        # target position
        self.target_position = np.r_[0.8 * self.maxp, 0.5 * self.maxp]

        self.low = np.r_[self.minp,
                         self.minp,
                         -self.maxv,
                         -self.maxv,
                         self.minp,
                         self.minp,
                         -self.maxv,
                         -self.maxv,
                         -np.inf]
        self.high = np.r_[self.maxp,
                          self.maxp,
                          +self.maxv,
                          +self.maxv,
                          self.maxp,
                          self.maxp,
                          +self.maxv,
                          +self.maxv,
                          np.inf]

        # parameters for the simulation -----------------------------
        self.velocity_increment = 0.02 # how fast to accelerate [m/s]* self.control_hz
        self.deacceleration = 0.4 # how fast to deaccelerate [m/s]* self.control_hz
        self.control_hz = 20.
        self.sim_hz = 1000.
        self.max_force = 400 # force of manipulator

        # parameters of the distribution
        self.mass_mean = 0.025
        self.mass_stdv = 0.005

        self.friction_mean = 0.5
        self.friction_sdv = 0.05

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
            [0.71, 0.71],
            [0.71, -0.71]
        ]) * self.velocity_increment

        self.reset()

    def reset(self):
        ''''''
        # reset simulation
        p.resetSimulation()
        p.setGravity(0.0, 0.0, -10.0)

        # load environment
        p.loadURDF("plane.urdf")

        # displacement
        displacement = (0.2+ np.random.rand()* 0.6) * self.maxp

        # load manipulation object -----------------------------------
        rot = Rotation.from_rotvec(np.r_[0.0, 1.0, 0.0] * 0.5 * np.pi)
        pos = [0.065, displacement, 0.01]
        self.object_id = p.loadURDF("urdfs/cuboid1.urdf", pos, rot.as_quat())
        # task randomization
        p.changeDynamics(self.object_id, -1, mass=np.random.normal(self.mass_mean, self.mass_stdv))
        p.changeDynamics(self.object_id, -1, lateralFriction=np.random.normal(self.friction_mean, self.friction_sdv))
        # velocity
        self.velocity_vector = np.zeros(2)

        # load manipulator ------------------------------------------
        # position
        self.robot_position = np.r_[0.03, displacement+ 0.02*(-0.5 + np.random.rand()), 0.008]
        rot = Rotation.from_rotvec(np.r_[1.0, 0.0, 0.0] * 0.5 * np.pi)
        self.orientation = rot.as_quat()
        self.robot_id = p.loadURDF("urdfs/end_effector1.urdf", self.robot_position,  self.orientation)

        # define constraint to track motion
        self.constraint_uid = p.createConstraint(self.robot_id, -1, -1, -1, p.JOINT_FIXED, [0.0, 0.0, 0.0],
                                                 [0., 0., 0.], self.robot_position, [0., 0., 0., 1.0], rot.as_quat())

        # return observation -----------------------------------------
        obs = np.concatenate([self._get_obs(), np.array([0])])
        return obs

    def step(self, action):
        ''''''
        # number of steps (control frequency < simulation frequency)
        N_steps = int(self.sim_hz / self.control_hz)

        # action
        if np.linalg.norm(self.robot_position[:2] - np.array([0.0, 0.5 * self.maxp])) < self.arm_length:
            # accelerate
            dvelocity = self.velocities[action]/ N_steps
        else:
            # deacceleration
            velocity_magnitude = np.linalg.norm(self.velocity_vector)+ 1e-6  # >= 0
            dvelocity = - np.min([1., self.deacceleration/ velocity_magnitude])* self.velocity_vector/ N_steps

        # object velocity
        forces = []

        # continuously track motion
        for _ in range(N_steps):
            self.velocity_vector += dvelocity
            self.robot_position[:2] += self.velocity_vector/ self.sim_hz
            p.changeConstraint(self.constraint_uid,
                               jointChildPivot=self.robot_position,
                               jointChildFrameOrientation=self.orientation,
                               maxForce=self.max_force)
            p.stepSimulation()
            #time.sleep(1./240)

            # get forces and torques
            forces.append(p.getConstraintState(self.constraint_uid))

        # extract contact forces
        mean_forces = np.mean(np.asarray(forces), axis=0)
        contact_force = np.linalg.norm(mean_forces[:2])
        #print(np.round(contact_force, 4))

        # get observation
        obs = np.concatenate([self._get_obs(), np.array([contact_force])])

        # check domain violations
        if np.any(obs < self.low) or np.any(obs > self.high):
            done = True
        else:
            done = False

        # if robot moves too far away from object
        if np.linalg.norm(obs[:2]- obs[4:6]) > 0.1:
            done = True

        # calculate reward
        reward = .01 / (.01 + np.linalg.norm(obs[4:6] - self.target_position)) # 1/ dist(object_to_target)
        #reward = 10.* (0.25- np.linalg.norm(obs[4:6] - self.target_position))

        return obs, reward, done, {}

    def _get_obs(self):
        # get object position
        pos_robot, _ = p.getBasePositionAndOrientation(self.robot_id)
        pos_object, _ = p.getBasePositionAndOrientation(self.object_id)
        vel_object, _ = p.getBaseVelocity(self.object_id)
        return np.concatenate([np.asarray(pos_robot[:2]),
                               np.asarray(self.velocity_vector),
                               np.asarray(pos_object[:2]),
                               np.asarray(vel_object[:2])])


if __name__ == '__main__':
    # for some basic tests
    env = PushEnv()
    env.reset()
    done = False
    step = 0

    track_arm = []

    while not done:
        # action = np.random.uniform(0.0, 1.0, (5,))
        action = [3 if step < 50 else 2][0]
        obs, reward, _, _ = env.step(action)
        step += 1
        track_arm.append(obs[:2])

        print(np.linalg.norm(env.velocity_vector))
        print(np.linalg.norm(obs[2:4]))

    track_arm = np.asarray(track_arm)
    plt.figure()
    plt.scatter(track_arm[:,0], track_arm[:,1])
    plt.show()