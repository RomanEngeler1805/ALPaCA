import numpy as np

class environment():
    ''' 7x1 square environment with reward sampled at random in corners '''

    def __init__(self, state_space):
        # size of square
        self.size = state_space

        # initialize agent
        self.state = np.zeros([self.size])
        self.state[(self.size-1)/2] = 1

        # termination flag
        self.d = 0

        # initialize target
        #self.target = np.array((self.size-1) * np.random.randint(0, 2))
        self.target = self.size - 1

    def _sample_env(self):
        ''' resample delta '''
        #self.target = np.array((self.size-1) * np.random.randint(0, 2))
        self.target = self.size- 1

    def _sample_state(self):
        self.state = np.zeros([self.size])
        self.state[(self.size - 1) / 2] = 1
        return self.state

    def reward(self):
        ''' reward function '''

        if self.state[self.target] == 1:
            return 1

        return 0#(self.size- 1)- np.abs(self.target- np.argmax(self.state))

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

        if not d:
            if action == 1 and pos != 0: # left
                self.state[pos] = 0
                self.state[pos- 1] = 1
            if action == 2 and pos != self.size-1: # down
                self.state[pos] = 0
                self.state[pos+ 1] = 1

        # update reward
        r = self.reward()

        # stack observation
        obs = np.array([self.state, r, d])

        return obs