import numpy as np

class bandit_environment():
    ''' contextual bandit '''

    def __init__(self):
        ''' r(s,a) = theta* phi(s,a) '''
        np.random.seed(1234)

        # state dimensions
        self.n_dim = 1

        # distribution of theta
        self.L = 1. * np.eye(self.n_dim)  # cholesky of variance (Var= L^T L)
        self.mu = 3. * np.ones([self.n_dim])

        # draw new theta
        self.theta = self.mu + np.matmul(self.L, np.random.normal(size=self.n_dim)) # gaussian

        # features
        self.state = np.array([np.random.rand()])

    def _sample_env(self):
        ''' resample theta '''
        self.theta = self.mu + np.matmul(self.L, np.random.normal(size=self.n_dim)) # gaussian

    def _sample_state(self):
        ''' resample state '''
        self.state = np.array([np.random.rand()])
        return self.state


    def _psi(self, state):
        ''' used to query environment '''
        state = state.reshape(-1, 1)

        psi = np.concatenate([np.sin(4.2 * np.pi * state - 0.3),
                              np.sin(4.2 * np.pi * state - 0.3 - np.pi / 2.),
                              np.sin(4.2 * np.pi * state - 0.3 - np.pi)], axis=1)
        return psi

    def _step(self, action):
        ''' interact with environment and return observation [s', r, d] '''

        psi = self._psi(self.state)[0]

        r = np.dot(self.theta, psi[action])

        d = 0

        # resample random data point
        self._sample_state()

        # observe (untransformed) state and reward
        obs = np.array([self.state.flatten(), r, d])

        return obs