import numpy as np

class bandit_environment():
    ''' contextual bandit '''

    def __init__(self):
        ''' r(s,a) = theta* phi(s,a) '''
        np.random.seed(1234)

        # state dimensions
        self.n_dim = 1
        # number of bandits
        self.n_bandits = 100

        # distribution of theta
        self.L = 1. * np.eye(self.n_dim)  # cholesky of variance (Var= L^T L)
        self.mu = 3. * np.ones([self.n_dim])

        # draw new theta
        self.theta = self.mu + np.matmul(self.L, np.random.normal(size=self.n_dim)) # gaussian

        # features
        self.state = np.linspace(0., 1., self.n_bandits).reshape(-1,1) # 1D array

        self.psi = np.concatenate([np.sin(4.1* np.pi* self.state- 0.2),
                                    np.sin(4.1* np.pi* self.state- 0.2)], axis= 1)

        # initial state
        self.bandit = 0

    def sample_env(self):
        ''' resample theta '''
        self.theta = self.mu + np.matmul(self.L, np.random.normal(size=self.n_dim)) # gaussian

    def sample_state(self):
        ''' resample state '''
        self.bandit = np.random.randint(self.n_bandits) # resample random data point
        return self.state[self.bandit]

    def step(self, action):
        ''' interact with environment and return observation [s', r, d] '''

        r = np.dot(self.theta, self.psi[self.bandit, action])

        d = 0

        # resample random data point
        self.bandit = np.random.randint(self.n_bandits)

        # observe (untransformed) state and reward
        obs = np.array([self.state[self.bandit].flatten(), r, d])

        return obs