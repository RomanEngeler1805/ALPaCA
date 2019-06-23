import numpy as np

class bandit_environment():
    ''' contextual bandit '''

    def __init__(self, action_space):
        ''' r(s,a) = theta* phi(s,a) '''
        np.random.seed(1234)

        # state dimensions
        self.n_dim = 1

        # action dimensions
        self.action_dim = action_space

        # distribution of theta
        self.L = 1. * np.eye(self.action_dim)  # cholesky of variance (Var= L^T L)
        self.mu = 3. * np.ones([self.action_dim])

        # draw new amplitude
        self.theta = self.mu + np.matmul(self.L, np.random.normal(size=self.action_dim)) # gaussian

        # draw new phase
        self.phase = np.random.rand(self.action_dim)* np.pi/ 1.

        # features
        self.state = np.array([np.random.rand()])

    def _sample_env(self):
        ''' resample theta '''
        self.theta = self.mu + np.matmul(self.L, np.random.normal(size=self.action_dim)) # gaussian
        self.phase = np.random.rand(self.action_dim)* np.pi/ 1.

    def _sample_state(self):
        ''' resample state '''
        self.state = np.array([np.random.rand()])
        return self.state

    def _psi(self, state, phase):
        ''' used to query environment '''
        state = state.reshape(-1, 1)

        psi = np.empty([len(state), 0])
        for a in range(self.action_dim):
            psi = np.concatenate([psi, np.sin(4. * np.pi * state + phase[a])], axis=1)

        return psi

    def _step(self, action):
        ''' interact with environment and return observation [s', r, d] '''

        # get encoding
        psi = self._psi(self.state, self.phase)[0]

        # randomly perturbed reward
        r = np.dot(self.theta[action], psi[action])#+ 0.22* np.random.normal()

        d = 0

        # resample random data point
        self._sample_state()

        # observe (untransformed) state and reward
        obs = np.array([self.state.flatten(), r, d])

        return obs
