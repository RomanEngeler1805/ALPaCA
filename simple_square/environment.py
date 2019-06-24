import numpy as np

class environment():
    ''' 7x1 square environment with reward sampled at random in corners '''

    def __init__(self):
        # size of square
        self.size = 7

        # initialize agent
        self.state = np.zeros([self.size])
        self.state[(self.size-1)/2] = 1

        # termination flag
        self.d = 0

        # initialize target
        self.target = np.array((self.size-1) * np.random.randint(0, 2))

    def _sample_env(self):
        ''' resample delta '''
        self.state = np.zeros([self.size])
        self.state[(self.size - 1) / 2] = 1

        self.target = np.array((self.size-1) * np.random.randint(0, 2))


    def reward(self):
        ''' reward function '''
        if self.state[self.target] == 1:
            return 1
        return 0

    def termination(self):
        ''' determine termination of MDP '''
        return self.d

    def _step(self, action):
        '''
        interact with environment and return observation [s', r, d]
        '''

        # update position
        pos = np.where(self.state == 1)[0]

        # update termination flag
        d = 0

        # update reward
        r = self.reward()

        if not d:
            if action == 1 and pos != 0: # left
                self.state[pos] = 0
                self.state[pos- 1] = 1
            if action == 2 and pos != self.size-1: # down
                self.state[pos] = 0
                self.state[pos+ 1] = 1

        # stack observation
        obs = np.array([self.state, r, d])

        return obs