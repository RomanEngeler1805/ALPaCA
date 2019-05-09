class bandit_environment():
    ''' contextual bandit '''

    def __init__(self):
        ''' r(s,a) = theta* phi(s,a) '''
        # distribution of theta
        self.mu = 1.0
        self.cov = 1.0

        # state dimensions
        self.n_dim = 5
        # number of bandits
        self.n_bandits = 10
        # number of actions
        self.n_actions = 4

        # draw new theta
        self.theta = np.random.rand(self.n_dim) * self.cov + self.mu

        # features
        self.state = np.random.rand(self.n_bandits, self.n_dim, self.n_actions)
        self.psi = np.sin(2 * np.pi * self.state)

        # initial state
        self.bandit = 0

    def reset(self):
        ''' reset the environment to its initial state '''
        self.__init__()

    def reward(self):
        ''' reward function '''
        return 0

    def get_bandit(self):
        ''' resample bandit '''
        self.bandit = np.random.randint(self.n_bandits)

    def step(self, action):
        '''
        interact with environment and return observation [s', r, d]
        '''

        r = np.dot(self.psi[self.bandit, :, action], self.theta)

        d = 0

        # observe (untransformed) state and reward
        obs = np.array([self.state[self.bandit, :, action].flatten(), r, d])

        return obs