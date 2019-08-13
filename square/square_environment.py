import numpy as np

class square_environment():
    ''' NxN square environment with reward in corners '''

    def __init__(self, state_space):
        # size of square
        self.size = 5

        # initialize agent
        self.state = np.array([(self.size-1)/2, (self.size-1)/2])

        # termination flag
        self.d = 0

        # initialize target
        target_y = (self.size-1) * np.random.randint(0, 2)
        target_x = 2 # (self.size-1) * np.random.randint(0, 2)
        self.target = np.array([target_y, target_x])

    def _sample_env(self):
        ''' resample delta '''
        target_y = (self.size - 1) * np.random.randint(0, 2)
        target_x = 2  # (self.size-1) * np.random.randint(0, 2)
        self.target = np.array([target_y, target_x])

    def _sample_state(self):
        self.state = np.array([(self.size-1)/2, (self.size-1)/2])
        return self.state.flatten()

    def reward(self):
        ''' reward function '''
        if self.state[0] == self.target[0] and self.state[1] == self.target[1]:
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
            if action == 1 and self.state[0] != self.size-1: # down
                self.state[0]+= 1
            if action == 2 and self.state[0] != 0: # up
                self.state[0] -= 1

        # stack observation
        obs = np.array([self.state.flatten(), r, d])

        return obs