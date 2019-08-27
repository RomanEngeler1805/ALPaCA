import numpy as np

class environment():
    ''' Grid wold consisting of N squares with reward sampled at random in corners '''

    def __init__(self, state_space):
        # size of square
        self.size = state_space

        # initialize agent
        self.state = np.zeros([self.size])
        self.state[0] = 1

        # termination flag
        self.d = 0

        # initialize target
        #self.target = np.array((self.size-1) * np.random.randint(0, 2))
        #self.target = self.size - 1
        self.target = (self.size-1) //2 * np.random.randint(1,3)

    def _sample_env(self):
        ''' resample delta '''
        #self.target = np.array((self.size-1) * np.random.randint(0, 2))
        #self.target = self.size- 1
        self.target = (self.size - 1) // 2 * np.random.randint(1, 3)

    def _sample_state(self):
        self.state = np.zeros([self.size])
        self.state[0] = 1

        thermo = np.zeros([self.size])
        thermo[0: np.where(self.state == 1)[0][0]+1] = 1
        return thermo

    def reward(self):
        ''' reward function '''
        # 1- 1.*np.abs(self.target- np.argmax(self.state))/self.size
        # (self.size- 1)- np.abs(self.target- np.argmax(self.state))
        #if self.state[self.target] == 1:
        #    return 1
        return 1.- 1.*np.abs(self.target- np.argmax(self.state))/self.size

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
            if action == 2 and pos != self.size-1: # right
                self.state[pos] = 0
                self.state[pos+ 1] = 1

        # update reward
        r = self.reward()

        # stack observation
        thermo = np.zeros([self.size])
        thermo[0: np.where(self.state == 1)[0][0]+1] = 1

        obs = np.array([thermo, r, d])

        return obs