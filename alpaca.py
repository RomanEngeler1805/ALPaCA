'''
Class structure for ALPaCA meta-RL code
'''

import numpy as np
import tensorflow as tf
import sys
import random
from collections import Counter
import os
import time
import logging

from replay_buffer import replay_buffer
from square_environment import square_environment


# General Hyperparameters
tf.flags.DEFINE_integer("batch_size", 10, "Batch size for training")
tf.flags.DEFINE_integer("action_space", 4, "Dimensionality of action space")
tf.flags.DEFINE_integer("state_space", 9, "Dimensionality of state space")
tf.flags.DEFINE_integer("hidden_space", 16, "Dimensionality of hidden space")
tf.flags.DEFINE_float("gamma", 0.8, "Discount factor")
tf.flags.DEFINE_float("learning_rate", 0.001, "Initial learning rate")
tf.flags.DEFINE_float("noise_variance", 0.1, "Noise variance")
tf.flags.DEFINE_integer("N_episodes", 500, "Number of episodes")
tf.flags.DEFINE_integer("N_tasks", 10, "Number of tasks")
tf.flags.DEFINE_integer("L_episode", 30, "Length of episodes")
tf.flags.DEFINE_integer("replay_memory_size", 100, "Size of replay memory")
tf.flags.DEFINE_integer("update_freq", 1, "Update frequency of posterior and sampling of new policy")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)

class QNetwork():
    def __init__(self, scope="QNetwork", summaries_dir="./"):
        self.gamma = FLAGS.gamma
        self.action_dim = FLAGS.action_space
        self.state_dim = FLAGS.state_space
        self.hidden_dim = FLAGS.hidden_space
        self.lr = FLAGS.learning_rate
        self.noise_variance = FLAGS.noise_variance

        with tf.variable_scope(scope):
            # build graph
            self._build_model()

        '''
            # summaries writer
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries")#_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
            self.summary_writer = tf.summary.FileWriter(summary_dir)

        '''

    def model(self, x, name="latent"):


        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # model architecture
            hidden1 = tf.contrib.layers.fully_connected(x, num_outputs=self.hidden_dim, activation_fn=tf.nn.relu)

            # TODO: what dimensions should hidden layer have? [batch, latent space, actions]
            #hidden2 = tf.contrib.layers.fully_connected(hidden1, num_outputs=self.hidden_dim, activation_fn=tf.nn.relu)

            hidden2 = tf.contrib.layers.fully_connected(hidden1, num_outputs=self.hidden_dim* self.action_dim, activation_fn=tf.nn.relu)
            hidden2 = tf.reshape(hidden2, [-1, self.hidden_dim, self.action_dim])

        return hidden2

    def predict(self, phi, name="pred"):
        ''' predict Q-values of  next time step '''

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # TODO: matmul of latent space with weights to get output. What about bias?
            Wout = tf.get_variable('wout', shape=[1, self.hidden_dim, 1], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            W = tf.tile(Wout, [1, 1, self.action_dim])  # [input, latent space, action space, output space]
            output = tf.nn.relu(tf.einsum('ijk,mjk->ik', phi, W))

            #output = tf.contrib.layers.fully_connected(phi, num_outputs=self.action_dim, activation_fn=None)

        return output


    def _build_model(self):
        ''' constructing tensorflow model '''
        # placeholders
        self.x = tf.placeholder(shape=[None, self.state_dim], dtype = tf.float32, name='x')
        self.phi = self.model(self.x) # latent space

        self.x_next = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name='x_next')
        self.phi_next = self.model(self.x_next)  # latent space

        self.reward = tf.placeholder(shape=[None], dtype=tf.float32, name='reward')
        self.termination = tf.placeholder(shape=[None], dtype=tf.float32, name='termination')

        # output layer (Bayesian)
        # context
        # self.context_x = tf.placeholder(tf.int32, shape=[None, self.state_dim], name="cx")
        # self.context_R = tf.placeholder(tf.float32, shape=[None], name="cR")

        # self.context_phi = self.model(self.context_x) # latent space

        # noise variance (yet random)
        #self.nvar = tf.get_variable("noise_variance", initializer=self.noise_variance)
        #self.Sigma_e = self.nvar* tf.eye(self.hidden_dim, name='noise_variance')

        # prior (updated via GD)
        #self.w0_bar = tf.get_variable('w_init', shape=[self.hidden_dim, 1], dtype=tf.float32, initializer=tf.contib.layers.xavier_initializer())
        #self.L_asym = tf.get_variable('L_asym', shape=[self.hidden_dim, self.hidden_dim], dtype=tf.float32, initializer=tf.contib.layers.xavier_initializer())  # cholesky decomp of \Lambda_0
        #self.L = self.L_asym @ tf.transpose(self.L_asym)  # \Lambda_0

        # inner loop: updating posterior distribution ========================================
        # posterior
        #self.wt_bar_inv = tf.matrix_inverse(tf.transpose(self.context_x)) @ self.context_x+ self.L
        #self.wt_bar = self.wt_bar_inv @ (tf.transpose(self.context_x) @ self.context_R + self.L @ self.w0_bar)


        # outer loop: updating network and prior distribution ================================
        # Sigma_pred = (1 + tf.matmul(tf.matmul(tf.transpose(self.phi), self.Kt_inv), self.phi)) * self.Sigma_e

        # predict Q-value
        self.Qout = self.predict(self.phi)

        # action placeholder
        self.action = tf.placeholder(shape=[None], dtype=tf.int32, name='action')
        self.actions_onehot = tf.one_hot(self.action, self.action_dim, dtype=tf.float32)
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        #
        Qmax = self.predict(self.phi_next)
        max_action = tf.one_hot(tf.argmax(Qmax, axis=1), self.action_dim, dtype=tf.float32)
        Qmax = tf.reduce_sum(tf.multiply(Qmax, max_action), axis=1)
        # last factor to account for case that s is terminating state
        self.Qtarget = self.reward+ FLAGS.gamma* tf.multiply(Qmax, 1-self.termination)

        # TODO: perform these operations for a batch
        # loss function
        # diff = self.Q_max- self.Q
        # logdet_Sigma = tf.linalg.logdet(Sigma_pred)
        # self.loss = tf.reduce_mean(tf.matmul(tf.matmul(diff, tf.matrix_inverse(Sigma_pred)), diff)+ logdet_Sigma)
        self.loss = tf.losses.mean_squared_error(self.Qtarget, self.Q)

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.updateModel = self.optimizer.apply_gradients(grads_and_vars)

        # summary
        loss_summary = tf.summary.scalar('loss', self.loss)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("/grad/hist/%s" % v.name, g)
                grad_summaries.append(grad_hist_summary)

        self.summaries_merged = tf.summary.merge([grad_summaries, loss_summary])


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

# epsilon-greedy
eps = 0.9
deps = (eps- 0.1)/ FLAGS.N_episodes

# get TF logger
log = logging.getLogger('Train')
log.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create file handler which logs even debug messages
logger_dir = './logger/'
if logger_dir:
    if not os.path.exists(logger_dir):
        os.makedirs(logger_dir)

fh = logging.FileHandler(logger_dir+'tensorflow_'+ time.strftime('%y-%m-%d_%H-%M')+ '.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)

# write hyperparameters to logger
log.info('Parameters')
for key in FLAGS.__flags.keys():
    log.info('{}={}'.format(key, getattr(FLAGS, key)))
print("")

# initialize replay memory and model
rbuffer = replay_buffer(FLAGS.replay_memory_size) # large buffer to store all experience
tempbuffer = replay_buffer(FLAGS.L_episode) # buffer for episode
log.info('Build Tensorflow Graph')
QNet = QNetwork() # neural network

# initialize environment
env = square_environment()

# session
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # checkpoint and summaries
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(logdir=os.path.join('./', 'summaries/', time.strftime('%y-%m-%d_%H-%M')), graph=sess.graph)

    '''
    # populate replay memory
    print("Populating replay memory...")
    log.info('Populating replay memory')
    state = env.reset()

    for i in range(FLAGS.replay_memory_size):

        Qval = sess.run(QNet.Qout, feed_dict={QNet.x: state.reshape(1,-1)}) # evaluate Q-fcn for state
        action = eGreedyAction(Qval) # pick epsilon-greedy action
        next_state, reward, done = env.step(action) # observe environment
        rbuffer.add(state, action, reward, next_state, done) # store experience

        if done:
            state = env.reset()
        else:
            state = next_state
    '''

    print('Target position: ' + str(env.target) +' (vertical, horizontal)') # not yet changing
    global_index = 0 # counter
    step_count = [] # count steps required to accomplish task
    state_count = [] # count number of visitations of each state

    # ==========================================================
    # loop episodes
    print("Episodes...")
    log.info('Loop over episodes')
    for episode in range(FLAGS.N_episodes):

        # loop tasks
        for n in range(FLAGS.N_tasks):
            # initialize buffer
            tempbuffer.reset()

            # reset environment
            state = env.reset()

            # loop steps
            step = 0
            while step < FLAGS.L_episode:
                # take a step
                Qval = sess.run(QNet.Qout, feed_dict={QNet.x: state.reshape(1,-1)})
                action = eGreedyAction(Qval, eps)
                next_state, reward, done = env.step(action)

                # store memory
                new_experience = [state, action, reward, next_state, done]
                tempbuffer.add(new_experience)


                #if step % FLAGS.update_freq:
                    #TODO: used only for ALPaCA

                # update state, and counters
                state = next_state
                global_index += 1
                step += 1

                # check if s is final state
                if done:
                    break

            # append episode buffer to large buffer
            rbuffer.add(tempbuffer.buffer)

            # count steps per trajectory
            step_count.append(step)


        # sample trajectories from buffer
        for k in range(FLAGS.batch_size):
            trajectory = rbuffer.sample(1)[0] # single trajecotory

            traj_len = len(trajectory) # length of trajectory
            state_train = np.zeros((traj_len, FLAGS.state_space))
            reward_train = np.zeros((traj_len,))
            action_train = np.zeros((traj_len,))
            next_state_train = np.zeros((traj_len, FLAGS.state_space))
            done_train = np.zeros((traj_len,))

            for k, experience in enumerate(trajectory):
                # [state, action, reward, next_state, done]
                state_train[k] = experience[0]
                action_train[k] = experience[1]
                reward_train[k] = experience[2]
                next_state_train[k] = experience[3]
                done_train[k] = experience[4]

            # update posterior
            _, summaries_merged = sess.run([QNet.updateModel, QNet.summaries_merged],
                                            feed_dict={QNet.x: state_train, QNet.x_next: next_state_train,
                                            QNet.action: action_train, QNet.reward: reward_train,  QNet.termination: done_train})

            # update summary
            summary_writer.add_summary(summaries_merged, global_index)
            summary_writer.flush()


        # epsilon-greedy schedule
        eps -= deps

        # count state visitations
        st = np.zeros((step, FLAGS.state_space))
        for k, (s0, a, r, s1, d) in enumerate(tempbuffer.buffer):
            st[k] = s0
        state_count.append(np.sum(st, axis=0))

        # print to console
        if episode % 50 == 0:
            log.info('Episode {:4d}, #Training steps {:2.0f}, Epsilon {:2.2f}'.format(episode, np.mean(step_count[-20:]), eps))
            print('Episode %4.d, #Training step %2.d, Epsilon %2.2f' % (episode, np.mean(step_count[-1]), eps))
            print('State Count: ')
            print(state_count[-1].reshape(3,3))

            # state value
            Qprint = np.zeros([FLAGS.state_space])
            for i in range(FLAGS.state_space):
                ss = np.zeros([FLAGS.state_space]) # loop over one-hot encoding
                ss[i] = 1
                Qval = sess.run(QNet.Qout, feed_dict={QNet.x: ss.reshape(1, -1)}) # get Q-value of current state
                Qprint[i] = np.mean(Qval) # V = E[Q]
            print('V-Value: ')
            print(np.round(Qprint.reshape(3, 3), 3))


    # update model and prior

# log file
# parser file
# TF summaries
# save model to restore