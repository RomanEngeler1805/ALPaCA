import numpy as np

class square_environment():
    ''' NxN square environment with reward in corners '''

    def __init__(self, state_space):
        # size of square
        self.size = state_space

        # initialize agent
        self.state = np.zeros([self.size, self.size])
        self.state[(self.size-1)/2, (self.size-1)/2] = 1

        # termination flag
        self.d = 0

        # initialize target
        target_x = (self.size-1) * np.random.randint(0, 2)
        target_y = 2 # (self.size-1) * np.random.randint(0, 2)
        self.target = np.array([target_x, target_y])

    def _sample_env(self):
        ''' resample delta '''
        target_x = (self.size - 1) * np.random.randint(0, 2)
        target_y = 2  # (self.size-1) * np.random.randint(0, 2)
        self.target = np.array([target_x, target_y])

    def _sample_state(self):
        self.state = np.zeros([self.size, self.size])
        self.state[(self.size - 1) / 2, (self.size - 1) / 2] = 1
        return self.state.flatten()

    def reward(self):
        ''' reward function '''
        target_x, target_y = self.target
        if self.state[target_x,target_y] == 1:
            return 1
        else:
            return -0.1
        return 0

    def termination(self):
        ''' determine termination of MDP '''
        return self.d

    def _step(self, action):
        '''
        interact with environment and return observation [s', r, d]
        '''

        # update position
        posx, posy = np.where(self.state == 1)

        # update termination flag
        d = 0

        # update reward
        r = self.reward()

        '''
        if not d:
            if action == 1 and posy != 0: # left
                #self.state[posx, posy] = 0
                #self.state[posx, posy - 1] = 1
                self.state=self.state
            if action == 2 and posx != self.size-1: # down
                self.state[posx, posy] = 0
                self.state[posx + 1, posy] = 1
            if action == 3 and posy != self.size-1: # right
                #self.state[posx, posy] = 0
                #self.state[posx, posy + 1] = 1
                self.state=self.state
            if action == 4 and posx != 0: # up
                self.state[posx, posy] = 0
                self.state[posx - 1, posy] = 1
        '''

        if not d:
            if action == 1 and posx != self.size-1: # left
                self.state[posx, posy] = 0
                self.state[posx + 1, posy] = 1
            if action == 2 and posx != 0: # down
                self.state[posx, posy] = 0
                self.state[posx - 1, posy] = 1

        # stack observation
        obs = np.array([self.state.flatten(), r, d])

        return obs