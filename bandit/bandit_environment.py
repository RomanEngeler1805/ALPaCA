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

        # draw new amplitude
        self.theta = self.mu + np.matmul(self.L, np.random.normal(size=self.n_dim)) # gaussian

        # draw new phase
        self.phase = np.random.rand()* np.pi/2.

        # features
        self.state = np.array([np.random.rand()])

    def _sample_env(self):
        ''' resample theta '''
        self.theta = self.mu + np.matmul(self.L, np.random.normal(size=self.n_dim)) # gaussian
        self.phase = np.random.rand()* np.pi/2.

    def _sample_state(self):
        ''' resample state '''
        self.state = np.array([np.random.rand()])
        return self.state


    def _psi(self, state, phase):
        ''' used to query environment '''
        state = state.reshape(-1, 1)

        psi = np.sin(4.2 * np.pi * state + phase - np.pi* 1./4.)
        #np.concatenate([np.sin(4.2 * np.pi * state + phase[0] - np.pi* 1./4.),
        #                      np.sin(4.2 * np.pi * state + phase[1] - np.pi* 2./4.),
        #                      np.sin(4.2 * np.pi * state + phase[2] - np.pi* 3./4.)], axis=1)
        return psi

    def _step(self, action):
        ''' interact with environment and return observation [s', r, d] '''

        # get encoding
        psi = self._psi(self.state, self.phase)[0]

        # randomly perturbed reward
        r = np.dot(self.theta, psi[action])#+ 0.1* np.random.normal()

        d = 0

        # resample random data point
        self._sample_state()

        # observe (untransformed) state and reward
        obs = np.array([self.state.flatten(), r, d])

        return obs