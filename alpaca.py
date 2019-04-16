'''
Class structure for ALPaCA meta-RL code
'''


class replay_buffer():
    def __init__(self):

    def sample(self):
        ''' sample new batch from replay buffer '''

    def add(self):
        ''' add new experience to replay buffer '''


class QNetwork():
    def __init__(self):

    def _build_model(self):
        ''' constructing tensorflow model '''

    def predict(self):
        ''' predict next time step '''

    def update_prior(self):
        ''' update prior parameters (w_bar_0, Sigma_0, Theta) via GD '''

    def update_posterior(self):
        ''' update posterior parameters (w_bar_t, Sigma_t) via closed-form expressions '''

    def Thompson(self):
        ''' resample weights from posterior '''

    def eGreedyAction(self):
        ''' select next action according to epsilon-greedy algorithm '''

    def Boltzmann(self):
        ''' select next action according to Boltzmann '''

    def GreedyAction(self):
        ''' select next action greedily '''


def copy_model_parameters():
    ''' copy weights from one model to another '''


class square_environment():
    def __init__(self):

    def reset(self):
        ''' reset the environment to its initial state '''

    def reward(self):
        ''' reward function '''

    def termination(self):
        ''' determine termination of MDP '''

    def step(self):
        ''' interact with environment and return observation [s', r, d] '''

# log file
# parser file
# TF summaries
# save model to restore