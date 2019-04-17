'''
Class structure for ALPaCA meta-RL code
'''

import numpy as np
import tensorflow as tf


class replay_buffer():
    def __init__(self):

    def sample(self):
        ''' sample new batch from replay buffer '''

    def add(self):
        ''' add new experience to replay buffer '''


class QNetwork():
    def __init__(self, scope="QNetwork", summaries_dir="summaries"):
        self.gamma = 0.99
        self.strategy = "epsilon_greedy"

        with tf.variable_scope(scope):
            # build graph
            self._build_model()

            # summaries writer
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
            self.summary_writer = tf.summary.FileWriter(summary_dir)


    def model(self, x, name="latent"):

        with tf.variable_scope(name, reuse=tf.AUTOREUSE):
            # model architecture
            W1 = tf.get_variable('weight_hidden1', shape=[9, 32], dtype=tf.float32)
            b1 = tf.get_variable('bias_hidden1', shape=[1, 32], dtype=tf.float32)
            hidden1 = tf.matmul(x, W1) + b1
            hidden1 = tf.nn.relu(hidden1)

            # TODO: what dimensions should hidden layer have? [input, latent space, actions]
            W2 = tf.get_variable('weight_hidden2', shape=[32, 32, 4], dtype=tf.float32)
            b2 = tf.get_variable('bias_hidden2', shape=[1, 32, 4], dtype=tf.float32)
            hidden2 = tf.matmul(hidden1, W2) + b2
            hidden2 = tf.nn.relu(hidden2)

        return hidden2


    def _build_model(self):
        ''' constructing tensorflow model '''
        # placeholders
        self.x = tf.placeholder(shape=[None, 9], dtype = tf.int32, name='x')
        self.phi = self.model(self.x) # latent space

        # action placeholder
        self.a = tf.placeholder(shape=[None, 4], dtype=tf.int32, name='a')

        # output layer (Bayesian)
        # context
        self.context_x = tf.placeholder(tf.int32, shape=[None, 9], name="cx")
        self.context_R = tf.placeholder(tf.float32, shape=[None], name="cy")

        self.context_phi = self.model(self.context_x) # latent space

        # noise variance (yet random)
        self.Sigma_e = 0.1* tf.eye(32, name='noise_variance')

        # prior
        self.K = tf.get_variable('K_init', shape=[32, 1], dtype=tf.float32)
        self.L_asym = tf.get_variable('L_asym', shape=[32, 32], dtype=tf.float32)  # cholesky decomp of \Lambda_0
        self.L = self.L_asym @ tf.transpose(self.L_asym)  # \Lambda_0

        # inner loop: updating posterior distribution ========================================
        # posterior
        self.Kt_inv = tf.matrix_inverse(tf.transpose(self.context_x)) @ self.context_x+ self.L
        self.Kt = self.Kt_inv @ (tf.transpose(self.context_x) @ self.context_R + self.L @ self.K)


        # outer loop: updating network and prior distribution ================================
        Sigma_pred = (1 + tf.matmul(tf.matmul(tf.transpose(self.phi), self.Kt_inv), self.phi)) * self.Sigma_e

        # TODO: need output of Qnetwork [None, 4]
        # predict Q-value



        if self.strategy == "epsilon_greedy":
            self.Q, self.prediction = self.eGreedyAction(self.phi) #?

        self.Q_max = tf.placeholder(tf.float32, shape=[None], name='qmax')

        # TODO: how to perform Q(s', argmax_a Q(s',a))
        # loss function
        diff = self.Q_max- self.Q
        logdet_Sigma = tf.linalg.log(tf.linalg.det(Sigma_pred))
        self.loss = tf.matmul(tf.matmul(diff, tf.matrix_inverse(Sigma_pred)), diff)+ logdet_Sigma



    def predict(self):
        ''' predict next time step '''

    def update_prior(self):
        ''' update prior parameters (w_bar_0, Sigma_0, Theta) via GD '''

    def update_posterior(self):
        ''' update posterior parameters (w_bar_t, Sigma_t) via closed-form expressions '''

    def Thompson(self):
        ''' resample weights from posterior '''

    def eGreedyAction(self, x, epsilon=0.9):
        ''' select next action according to epsilon-greedy algorithm '''

        # TODO: matmul of latent space with mean of prior to get output. What about bias?
        batch_size = tf.shape(x)[0]
        Wlast = tf.tile(self.K, multiples=[batch_size,4])
        Qlast = tf.matmul(x, Wlast)

        # epsilon-greedy


        if tf.random.uniform(shape=[], minval=0., maxval=1.) < epsilon:
            action = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
            Qout = tf.gather()

        return

    def Boltzmann(self):
        ''' select next action according to Boltzmann '''

    def GreedyAction(self):
        ''' select next action greedily '''


def copy_model_parameters(sess, estimator1, estimator2):
    """
    Copies the model parameters of one model to another.
    Args:
      sess: Tensorflow session instance
      estimator1: Estimator to copy the paramters from
      estimator2: Estimator to copy the parameters to
    """
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)



# log file
# parser file
# TF summaries
# save model to restore