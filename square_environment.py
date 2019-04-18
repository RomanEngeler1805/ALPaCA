import numpy as np

class square_environment():
    ''' 3x3 square environment with reward in corners '''

    def __init__(self):
        # size of square
        self.size = 3

        # initialize agent
        self.state = np.zeros([self.size, self.size])
        self.state[(self.size-1)/2, (self.size-1)/2] = 1

        # termination flag
        self.d = 0

        # initialize target
        target_x = 2#(self.size-1) * np.random.randint(0, 2)
        target_y = 0#(self.size-1) * np.random.randint(0, 2)
        self.target = np.array([target_x, target_y])

    def reset(self):
        ''' reset the environment to its initial state '''
        self.__init__()

        return self.state.flatten()

    def reward(self):
        ''' reward function '''
        target_x, target_y = self.target
        if self.state[target_x,target_y] == 1:
            self.d = 1
            return 1

        return 0

    def termination(self):
        ''' determine termination of MDP '''
        return self.d

    def step(self, action):
        '''
        interact with environment and return observation [s', r, d]
        '''

        # update position
        posx, posy = np.where(self.state == 1)

        # update reward
        r = self.reward()

        # update termination flag
        d = self.termination()

        if not d:
            if action == 0 and posy != 0:
                self.state[posx, posy] = 0
                self.state[posx, posy - 1] = 1
            if action == 1 and posx != self.size-1:
                self.state[posx, posy] = 0
                self.state[posx + 1, posy] = 1
            if action == 2 and posy != self.size-1:
                self.state[posx, posy] = 0
                self.state[posx, posy + 1] = 1
            if action == 3 and posx != 0:
                self.state[posx, posy] = 0
                self.state[posx - 1, posy] = 1


        # stack observation
        obs = np.array([self.state.flatten(), r, d])

        return obs