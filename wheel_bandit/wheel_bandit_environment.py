import numpy as np

class wheel_bandit_environment():
    ''' contextual bandit '''

    def __init__(self, action_space, random_seed):
        ''' r(s,a) = theta* phi(s,a) '''
        np.random.seed(1234)

        # state dimensions
        self.n_dim = 2

        # action dimensions
        self.action_dim = action_space

        # state
        self.state = np.random.rand(self.n_dim) # [radius, phase]

        # feature
        self.delta = (0.+ 0.99*np.random.rand())
        self.mu = np.array([1.2, 1.0, 50.]) # mean values for normal distributions
        self.sigma = 0.01 # standard deviation

    def _sample_env(self):
        ''' resample delta '''
        self.delta = (0.+ 0.99*np.random.rand())

    def _sample_state(self):
        ''' resample state '''
        self.state = np.random.rand(self.n_dim) # [radius, phase]
        return np.array([self.state[0]* np.cos(2.*np.pi* self.state[1]),
                                  self.state[0]* np.sin(2.*np.pi* self.state[1])])

    def _mu_idx(self, state):
        ''' used to calculate reward '''

        if state[0] > self.delta:
            if  0<= state[1] and state[1] < 0.25:
                mu_idx = np.array([0, 2, 1, 1, 1])
            elif 0.25 <= state[1] and state[1] < 0.50:
                mu_idx = np.array([0, 1, 2, 1, 1])
            elif 0.50 <= state[1] and state[1] < 0.75:
                mu_idx = np.array([0, 1, 1, 2, 1])
            elif 0.75 <= state[1] and state[1] <= 1.00:
                mu_idx = np.array([0, 1, 1, 1, 2])
        else:
            mu_idx = np.array([0, 1, 1, 1, 1])

        return mu_idx

    def _step(self, action):
        ''' interact with environment and return observation [s', r, d] '''

        # get encoding
        mu_idx = self._mu_idx(self.state)

        # reward
        r = self.mu[mu_idx[action]]+ self.sigma* np.random.normal()

        d = 0

        # resample random data point
        self._sample_state()

        # observe (untransformed) state and reward
        obs = np.array([np.array([self.state[0]* np.cos(2.*np.pi* self.state[1]),
                                  self.state[0]* np.sin(2.*np.pi* self.state[1])]), r, d])

        return obs
