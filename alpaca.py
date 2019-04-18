'''
Class structure for ALPaCA meta-RL code
'''

import numpy as np
import tensorflow as tf
import sys
import random
from collections import Counter
import os
from replay_buffer import replay_buffer
from square_environment import square_environment


# General Hyperparameters
tf.flags.DEFINE_integer("batch_size", 4, "Batch size for training")
tf.flags.DEFINE_integer("action_space", 4, "Dimensionality of action space")
tf.flags.DEFINE_integer("state_space", 9, "Dimensionality of state space")
tf.flags.DEFINE_integer("hidden_space", 16, "Dimensionality of hidden space")
tf.flags.DEFINE_float("gamma", 0.9, "Discount factor")
tf.flags.DEFINE_float("learning_rate", 0.9, "Initial learning rate")
tf.flags.DEFINE_float("noise_variance", 0.1, "Noise variance")
tf.flags.DEFINE_integer("N_episodes", 1000, "Number of episodes")
tf.flags.DEFINE_integer("L_episode", 30, "Length of episodes")
tf.flags.DEFINE_integer("replay_memory_size", 1000, "Size of replay memory")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")



class QNetwork():
    def __init__(self, scope="QNetwork", summaries_dir="summaries"):
        self.gamma = FLAGS.gamma
        self.action_dim = FLAGS.action_space
        self.state_dim = FLAGS.state_space
        self.hidden_dim = FLAGS.hidden_space
        self.lr = FLAGS.learning_rate
        self.nvar = FLAGS.noise_variance

        with tf.variable_scope(scope):
            # build graph
            self._build_model()

            # summaries writer
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries")#_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
            self.summary_writer = tf.summary.FileWriter(summary_dir)


    def model(self, x, name="latent"):


        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # model architecture
            hidden1 = tf.contrib.layers.fully_connected(x, num_outputs=self.hidden_dim, activation_fn=tf.nn.relu)

            # TODO: what dimensions should hidden layer have? [batch, latent space, actions]
            hidden2 = tf.contrib.layers.fully_connected(hidden1, num_outputs=self.hidden_dim* self.action_dim, activation_fn=tf.nn.relu)
            hidden2 = tf.reshape(hidden2, [-1, self.hidden_dim, self.action_dim])

        return hidden2

    def predict(self, phi, name="pred"):
        ''' predict Q-values of  next time step '''

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # TODO: matmul of latent space with weights to get output. What about bias?
            Wout = tf.get_variable('wout', shape=[1, self.hidden_dim, 1], dtype=tf.float32, )
            W = tf.tile(Wout, [1, 1, self.action_dim])  # [input, latent space, action space, output space]
            # TODO: is this correct? check flow in graph
            output = tf.nn.relu(tf.einsum('ijk,mjk->ik', phi, W))

        return output


    def _build_model(self):
        ''' constructing tensorflow model '''
        # placeholders
        self.x = tf.placeholder(shape=[None, self.state_dim], dtype = tf.float32, name='x')
        self.phi = self.model(self.x) # latent space

        # output layer (Bayesian)
        # context
        # self.context_x = tf.placeholder(tf.int32, shape=[None, self.state_dim], name="cx")
        # self.context_R = tf.placeholder(tf.float32, shape=[None], name="cR")

        # self.context_phi = self.model(self.context_x) # latent space

        # noise variance (yet random)
        # self.Sigma_e = self.nvar* tf.eye(self.hidden_dim, name='noise_variance')

        # prior
        # self.K = tf.get_variable('K_init', shape=[self.hidden_dim, 1], dtype=tf.float32)
        # self.L_asym = tf.get_variable('L_asym', shape=[self.hidden_dim, self.hidden_dim], dtype=tf.float32)  # cholesky decomp of \Lambda_0
        # self.L = self.L_asym @ tf.transpose(self.L_asym)  # \Lambda_0

        # inner loop: updating posterior distribution ========================================
        # posterior
        # self.Kt_inv = tf.matrix_inverse(tf.transpose(self.context_x)) @ self.context_x+ self.L
        # self.Kt = self.Kt_inv @ (tf.transpose(self.context_x) @ self.context_R + self.L @ self.K)


        # outer loop: updating network and prior distribution ================================
        # Sigma_pred = (1 + tf.matmul(tf.matmul(tf.transpose(self.phi), self.Kt_inv), self.phi)) * self.Sigma_e

        # predict Q-value
        self.Qout = self.predict(self.phi)
        # action placeholder
        self.action = tf.placeholder(shape=[None], dtype=tf.int32, name='action')
        self.actions_onehot = tf.one_hot(self.action, self.action_dim, dtype=tf.float32)
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        #
        self.Q_max = tf.placeholder(tf.float32, shape=[None], name='qmax') # R+ Q(s', argmax_a Q(s',a))

        # TODO: perform these operations for a batch
        # loss function
        # diff = self.Q_max- self.Q
        # logdet_Sigma = tf.linalg.logdet(Sigma_pred)
        # self.loss = tf.reduce_mean(tf.matmul(tf.matmul(diff, tf.matrix_inverse(Sigma_pred)), diff)+ logdet_Sigma)
        self.loss = tf.losses.mean_squared_error(self.Q_max, self.Q)

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.updateModel = self.optimizer.minimize(self.loss)

        # summary
        self.loss_summary = tf.summary.scalar('loss', self.loss)

    def update_prior(self):
        ''' update prior parameters (w_bar_0, Sigma_0, Theta) via GD '''

    def update_posterior(self):
        ''' update posterior parameters (w_bar_t, Sigma_t) via closed-form expressions '''

    def Thompson(self):
        ''' resample weights from posterior '''

    def eGreedyAction(self, x, epsilon=0.9):
        ''' select next action according to epsilon-greedy algorithm '''

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


def eGreedyAction(x, epsilon=0.9):
    ''' select next action according to epsilon-greedy algorithm '''

    if np.random.rand() > epsilon:
        action = np.argmax(x)
    else:
        action = np.random.randint(FLAGS.action_space)

    return action


# ==========================================================================================

# variables
eps = 0.9
deps = eps/ 1e3


# initialize replay memory and model
rbuffer = replay_buffer(FLAGS.replay_memory_size)
tempbuffer = replay_buffer(FLAGS.L_episode)
QNet = QNetwork()

# initialize environment
env = square_environment()

# session
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # checkpoint and summaries
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(logdir=os.path.join('./', 'summaries'), graph=sess.graph)

    # populate replay memory
    print("Populating replay memory...")
    state = env.reset()
    print('Target position: ' + str(env.target))

    for i in range(FLAGS.replay_memory_size):

        Qval = sess.run(QNet.Qout, feed_dict={QNet.x: state.reshape(1,-1)})
        action = eGreedyAction(Qval)
        next_state, reward, done = env.step(action)
        rbuffer.add(state, action, reward, next_state, done)

        if done:
            state = env.reset()
        else:
            state = next_state


    global_index = 0
    step_count = []
    state_count = []

    # loop episodes
    print("Episodes...")
    for episode in range(FLAGS.N_episodes):

        # reset environment
        state = env.reset()
        tempbuffer.reset()

        # loop steps
        step = 0
        while step < FLAGS.L_episode:
            # take a step
            Qval = sess.run(QNet.Qout, feed_dict={QNet.x: state.reshape(1,-1)})
            action = eGreedyAction(Qval, eps)
            next_state, reward, done = env.step(action)

            # TODO: possibly have additional buffer for current episode which is appended at the end to larger buffer
            # store memory
            tempbuffer.add(state, action, reward, next_state, done)

            # sample from larger buffer [s, a, r, s', d]
            experience = rbuffer.sample(FLAGS.batch_size)

            state_train = np.zeros((FLAGS.batch_size, FLAGS.state_space))
            reward_train = np.zeros((FLAGS.batch_size,))
            action_train = np.zeros((FLAGS.batch_size,))
            next_state_train = np.zeros((FLAGS.batch_size, FLAGS.state_space))
            done_train = np.zeros((FLAGS.batch_size,))

            for k, (s0, a, r, s1, d) in enumerate(experience):
                state_train[k] = s0
                reward_train[k] = r
                action_train[k] = a
                next_state_train[k] = s1
                done_train[k] = d

            # update posterior
            Qval = sess.run(QNet.Qout, feed_dict={QNet.x: state_train}) # feed with s' -> Q(s',a')
            Qmax = Qval[range(FLAGS.batch_size), np.argmax(Qval,axis=1)] # Q(s', argmax_a Q(s',a))

            # TODO: last factor to account for case that s' is terminating state -> necessary?
            Qtarget = reward_train+ FLAGS.gamma* Qmax* (1- done_train) # Q(s,a) = R + gamma* Q(s', argmax_a Q(s',a))

            _, loss_summary = sess.run([QNet.updateModel, QNet.loss_summary], feed_dict={QNet.x: state_train, QNet.Q_max: Qtarget, QNet.action: action_train})

            # update summary
            summary_writer.add_summary(loss_summary, global_index)
            #summary_op = tf.summary.merge([train_loss, train_loss_lin, train_loss_ang, train_learning_rate])
            summary_writer.flush()

            state = next_state
            global_index+=1
            step+= 1

            #
            if done:
                break

        # epsilon-greedy
        if eps >= 0.1:
            eps-= deps

        #
        step_count.append(step)

        # append buffer
        st = np.zeros((step, FLAGS.state_space))
        for k, (s0, a, r, s1, d) in enumerate(tempbuffer.buffer):
            st[k] = s0

        state_count.append(np.sum(st, axis=0))

        if episode % 50 == 0:
            print('Episode %4.d, #Training step %2.d, Epsilon %2.2f' % (episode, np.mean(step_count[-1]), eps))
            print(np.transpose(state_count[-1].reshape(3,3)))

        # append buffer
        for s0, a, r, s1, d in tempbuffer.buffer:
            rbuffer.add(s0, a, r, s1, d)

    print(eps)

    # update model and prior

# log file
# parser file
# TF summaries
# save model to restore