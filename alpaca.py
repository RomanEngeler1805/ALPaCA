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
tf.flags.DEFINE_float("noise_variance", 0.001, "Noise variance")
tf.flags.DEFINE_float("split", 0.5, "Split between fraction of trajectory used for updating posterior vs prior")
tf.flags.DEFINE_integer("N_episodes", 500, "Number of episodes")
tf.flags.DEFINE_integer("N_tasks", 5, "Number of tasks")
tf.flags.DEFINE_integer("L_episode", 10, "Length of episodes")
tf.flags.DEFINE_integer("replay_memory_size", 100, "Size of replay memory")
tf.flags.DEFINE_integer("update_freq", 1, "Update frequency of posterior and sampling of new policy")
tf.flags.DEFINE_integer("iter_amax", 5, "Number of iterations performed to determine amax")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)

class QNetwork():
    def __init__(self, scope="QNetwork"):
        self.gamma = FLAGS.gamma
        self.action_dim = FLAGS.action_space
        self.state_dim = FLAGS.state_space
        self.hidden_dim = FLAGS.hidden_space
        self.lr = FLAGS.learning_rate
        self.noise_variance = FLAGS.noise_variance
        self.iter_amax = FLAGS.iter_amax

        with tf.variable_scope(scope):
            # build graph
            self._build_model()

    def model(self, x, name="latent"):
        ''' Embedding into latent space '''
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # model architecture
            hidden1 = tf.contrib.layers.fully_connected(x, num_outputs=self.hidden_dim, activation_fn=tf.nn.relu)
            hidden2 = tf.contrib.layers.fully_connected(hidden1, num_outputs=self.hidden_dim* self.action_dim, activation_fn=tf.nn.relu)
            hidden2 = tf.reshape(hidden2, [-1, self.hidden_dim, self.action_dim])

        return hidden2

    def predict(self, phi, name="prediction"):
        ''' predict Q-values of  next time step '''
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            Wt = tf.get_variable('wt', shape=[1, self.hidden_dim, 1], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            W = tf.tile(Wt, [1, 1, self.action_dim])  # [input, latent space, action space, output space]
            output = tf.einsum('ijk,mjk->ik', phi, W)

        return output


    def _build_model(self):
        ''' constructing tensorflow model '''
        # placeholders ====================================================================
        self.state = tf.placeholder(shape=[None, self.state_dim], dtype = tf.float32, name='state') # input
        self.state_next = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name='next_state')  # input

        self.phi = self.model(self.state) # latent space
        self.phi_next = self.model(self.state_next)  # latent space

        self.action = tf.placeholder(shape=[None], dtype=tf.int32, name='action')
        self.done = tf.placeholder(shape=[None,1], dtype=tf.float32, name='done')

        self.reward = tf.placeholder(shape=[None], dtype=tf.float32, name='reward')

        # output layer (Bayesian) =========================================================
        # noise variance (scalar -> batchsize)
        self.nvar = self.noise_variance #tf.get_variable("noise_variance", initializer=self.noise_variance)
        (bs, _, _) = tf.unstack(tf.to_int32(tf.shape(self.phi)))
        self.Sigma_e = self.nvar * tf.ones(bs, name='noise_variance')

        self.wt = tf.get_variable('wt', shape=[self.hidden_dim])

        # prior (updated via GD) ---------------------------------------------------------
        with tf.variable_scope('prior', reuse=tf.AUTO_REUSE):
            self.w0_bar = tf.get_variable('w0_bar', dtype=tf.float32, shape=[self.hidden_dim,1])
            self.L0_asym = tf.get_variable('L0_asym', dtype=tf.float32, initializer=tf.eye(self.hidden_dim))  # cholesky decomp of \Lambda_0
            self.L0 = tf.matmul(self.L0_asym, tf.transpose(self.L0_asym))  # \Lambda_0

            self.sample_prior = self._sample_prior()

        # posterior (analytical update) --------------------------------------------------
        with tf.variable_scope('posterior', reuse=tf.AUTO_REUSE):
            # phi(s, a)
            taken_action = tf.one_hot(tf.reshape(self.action, [-1,1]), self.action_dim, dtype=tf.float32)
            phi_taken = tf.reduce_sum(tf.multiply(self.phi, taken_action), axis=2)

            # update posterior
            self.wt_bar, self.Lt_inv = self._max_posterior(self.phi_next, phi_taken, self.done, self.reward)

            # phi(s', a*)
            Q_next = self.predict(self.phi_next)
            max_action = tf.one_hot(tf.reshape(tf.argmax(Q_next, axis=1), [-1, 1]), self.action_dim, dtype=tf.float32)
            phi_max = tf.reduce_sum(tf.multiply(self.phi_next, max_action), axis=2)  #[batch_size, hidden_dim]
            phi_max = tf.stop_gradient(phi_max)

            # sample posterior distribution
            self.phi_hat = phi_taken- self.gamma* tf.multiply(phi_max, 1- self.done)

            self.sample_post = self._sample_posterior(tf.reshape(self.wt_bar, [-1,1]), self.Lt_inv)

        # predict Q-value
        self.Qout = self.predict(self.phi)

        # loss and summaries ==============================================================
        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            # The next three lines are included since for the GD update step, we require the posterior calculated by the
            # context information, which is different than the information used for calculating the gradient.
            # to avoid two more placeholders for the context data (and the resulting use of more phi's and Q's)
            # I currently solve it by interrupting the computational graph here
            self.max_post = self._max_posterior(self.phi_next, phi_taken, self.done, self.reward) # based on training data
            self.wt_bar_max = tf.placeholder(shape=[self.hidden_dim], dtype=tf.float32, name='wt_bar_max') # TODO
            self.Lt_inv_max = tf.placeholder(shape=[self.hidden_dim, self.hidden_dim], dtype=tf.float32, name='Lt_inv_max') # TODO

            self.Qdiff = self.reward+ tf.einsum('i,ji->j', self.wt_bar_max, -self.phi_hat) # [batch_size] / minus needs to be there

            # TODO: check this. can sigma be calculated with a single einsum? (would likely require use of delta-fcn)
            # stacked covariance (column vector)
            Sigma_pred = tf.einsum('ij,jk->ik', self.phi_hat, self.Lt_inv_max) # [batch_size,1]
            Sigma_pred = tf.reduce_sum(tf.multiply(Sigma_pred, self.phi_hat), axis=1) + self.Sigma_e
            logdet_Sigma = tf.reduce_sum(Sigma_pred) # logdet(sigma) with sigma: [batch_size,1]

            # loss function
            self.loss = tf.einsum('i,ik,k->', self.Qdiff, tf.linalg.inv(tf.linalg.diag(Sigma_pred)), self.Qdiff)+ logdet_Sigma

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.updateModel = self.optimizer.apply_gradients(grads_and_vars)

        # summary
        self.loss_summary = tf.summary.scalar('loss', self.loss)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("/grad/hist/%s" % v.name, g)
                grad_summaries.append(grad_hist_summary)

        self.summaries_merged = tf.summary.merge([grad_summaries, self.loss_summary])


    def _sample_prior(self):
        ''' sample wt from prior '''
        # with tf.variable_scope('sample_prior', reuse=tf.AUTO_REUSE):
        update_op = tf.assign(self.wt, self._sample_MN(self.w0_bar, tf.matrix_inverse(self.L0)))
        return update_op

    def _sample_posterior(self, wt_bar, Lt_inv):
        ''' sample wt from posterior '''
        #with tf.variable_scope('sample_posterior', reuse=tf.AUTO_REUSE):
        update_op = tf.assign(self.wt, self._sample_MN(wt_bar, Lt_inv))
        return update_op

    def _sample_MN(self, mu, cov):
        ''' sample from multi-variate normal '''
        A = tf.linalg.cholesky(cov)
        z = tf.random_normal(shape=[self.hidden_dim,1])
        return tf.reshape(mu+ tf.matmul(A,z), [-1])

    def _update_posterior(self, phi_hat, reward):
        ''' update posterior distribution '''
        Lt = tf.matmul(tf.transpose(phi_hat), phi_hat) + self.L0
        Lt_inv = tf.matrix_inverse(Lt)
        wt_unnormalized = tf.matmul(self.L0, self.w0_bar) + \
                          tf.matmul(tf.transpose(phi_hat), tf.reshape(reward, [-1, 1]))
        wt_bar = tf.matmul(Lt_inv, wt_unnormalized)

        return wt_bar, Lt_inv

    def _max_posterior(self, phi_next, phi_taken, done, reward):
        ''' determine wt_bar for calculating phi(s_{t+1}, a*) '''
        # determine phi(max_action) based on Q determined by sampling wt from prior
        _ = self._sample_prior() # sample wt
        Q_next = self.predict((phi_next))
        max_action = tf.one_hot(tf.reshape(tf.argmax(Q_next, axis=1), [-1, 1]), self.action_dim, dtype=tf.float32)
        phi_max = tf.reduce_sum(tf.multiply(phi_next, max_action), axis=2)

        # iterations
        for _ in range(self.iter_amax):
            # sample posterior
            phi_hat = phi_taken - self.gamma* tf.multiply(phi_max, 1 - done)

            # update posterior distributior
            wt_bar, Lt_inv = self._update_posterior(phi_hat, reward)

            # determine phi(max_action) based on Q determined by sampling wt from posterior
            _ = self._sample_posterior(wt_bar, Lt_inv)
            Q_next = self.predict((phi_next))
            max_action = tf.one_hot(tf.reshape(tf.argmax(Q_next, axis=1), [-1, 1]), self.action_dim, dtype=tf.float32)
            phi_max = tf.reduce_sum(tf.multiply(phi_next, max_action), axis=2)

        return tf.reshape(wt_bar, [-1]), Lt_inv

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


# Main Routine ===========================================================================

# epsilon-greedy
eps = 0.
#deps = (eps- 0.1)/ FLAGS.N_episodes

# get TF logger --------------------------------------------------------------------------
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

# initialize replay memory ----------------------------------------------------------------
fullbuffer = replay_buffer(FLAGS.replay_memory_size) # large buffer to store all experience
tempbuffer = replay_buffer(FLAGS.L_episode) # buffer for episode

# initialize model
log.info('Build Tensorflow Graph')
QNet = QNetwork()  # neural network

# initialize environment
env = square_environment()

with tf.Session() as sess:
    # session
    init = tf.global_variables_initializer()
    sess.run(init)

    # checkpoint and summaries
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(logdir=os.path.join('./', 'summaries/', time.strftime('%y-%m-%d_%H-%M')), graph=sess.graph)

    print('Target position: ' + str(env.target) +' (vertical, horizontal)') # not yet random sampling

    # initialize
    global_index = 0 # counter
    step_count = [] # count steps required to accomplish task

    # -----------------------------------------------------------------------------------
    # loop episodes
    print("Episodes...")
    log.info('Loop over episodes')
    for episode in range(FLAGS.N_episodes):

        # loop tasks --------------------------------------------------------------------
        for n in range(FLAGS.N_tasks):
            # initialize buffer
            tempbuffer.reset()
            # reset environment
            state = env.reset()
            # sample w from prior
            sess.run([QNet.sample_prior])

            # loop steps
            step = 0
            while step < FLAGS.L_episode:
                # take a step in the environment
                Qval = sess.run([QNet.Qout], feed_dict={QNet.state: state.reshape(1,-1)})

                action = eGreedyAction(Qval, eps)
                next_state, reward, done = env.step(action)

                # store experience in memory
                new_experience = [state, action, reward, next_state, done]
                tempbuffer.add(new_experience)

                # update posterior
                # TODO: could speed up by iteratively adding
                if step % FLAGS.update_freq:
                    reward_train = np.zeros([step,])
                    state_train = np.zeros([step, FLAGS.state_space])
                    next_state_train = np.zeros([step, FLAGS.state_space])
                    action_train = np.zeros([step, FLAGS.action_space])
                    done_train = np.zeros([step, 1])

                    for k, experience in enumerate(tempbuffer.buffer):
                        # [s, a, r, s', a*, d]
                        state_train[k] = experience[0]
                        action_train[k] = experience[1]
                        reward_train[k] = experience[2]
                        next_state_train[k] = experience[3]
                        done_train[k] = experience[4]

                    sess.run(QNet.sample_post, feed_dict={QNet.state: state_train, QNet.action: action_train,
                                                          QNet.reward: reward_train, QNet.state_next: next_state_train,
                                                          QNet.done: done_train})
                
                # update state, and counters
                state = next_state
                global_index += 1
                step += 1

                # check if s is final state
                # if done:
                #    break

            # append episode buffer to large buffer
            fullbuffer.add(tempbuffer.buffer)

            # count steps per trajectory
            step_count.append(step)

        # ---------------------------------------------------------------------------
        # sample trajectories from buffer
        # TODO: check if different trajectories are sampled
        updatebuffer = fullbuffer.sample(5)
        for trajectory in updatebuffer:
            traj_len = len(trajectory) # length of trajectory

            state_sample = np.zeros([traj_len, FLAGS.state_space])
            action_sample = np.zeros([traj_len,])
            reward_sample = np.zeros([traj_len, ])
            next_state_sample = np.zeros([traj_len, FLAGS.state_space])
            done_sample = np.zeros([traj_len,1])

            for k, experience in enumerate(trajectory):
                # [s, a, r, s', a*, d]
                state_sample[k] = experience[0]
                action_sample[k] = experience[1]
                reward_sample[k] = experience[2]
                next_state_sample[k] = experience[3]
                done_sample[k] = experience[4]

            # split in train and validation set
            train = np.random.choice(np.arange(traj_len), np.int(FLAGS.split* traj_len),replace=False) # mixed
            valid = np.setdiff1d(np.arange(traj_len), train)

            state_train = state_sample[train, :]
            action_train = action_sample[train]
            reward_train = reward_sample[train]
            next_state_train = next_state_sample[train, :]
            done_train = done_sample[train]

            state_valid = state_sample[valid, :]
            action_valid = action_sample[valid]
            reward_valid = reward_sample[valid]
            next_state_valid = next_state_sample[valid, :]
            done_valid = done_sample[valid]

            # update posterior
            wt, Lt_inv = sess.run(QNet.max_post, feed_dict={QNet.state: state_train, QNet.action: action_train,
                                                            QNet.reward: reward_train, QNet.state_next: next_state_train,
                                                            QNet.done: done_train})

            # update prior via GD
            _, summaries_merged= sess.run([QNet.updateModel, QNet.summaries_merged],
                                           feed_dict={QNet.state: state_valid, QNet.action: action_valid,
                                                      QNet.reward: reward_valid, QNet.state_next: next_state_valid,
                                                      QNet.done: done_valid,
                                                      QNet.wt_bar_max: wt, QNet.Lt_inv_max: Lt_inv})

            # update summary
            summary_writer.add_summary(summaries_merged, global_index)
            summary_writer.flush()

        # epsilon-greedy schedule
        #eps -= deps

        # output to console -------------------------------------------------------
        # count state visitations
        #st = np.zeros((step, FLAGS.state_space))
        #for k, experience in enumerate(tempbuffer.buffer):
        #    st[k] = experience[0]
        #state_count.append(np.sum(st, axis=0))

        # print to console
        if episode % 50 == 0:
            log.info('Episode {:4d}, #Training steps {:2.0f}, Epsilon {:2.2f}'.format(episode, np.mean(step_count[-20:]), eps))
            print('Episode %4.d, #Training step %2.d, Epsilon %2.2f' % (episode, np.mean(step_count[-1]), eps))
            #print('State Count: ')
            #print(state_count[-1].reshape(3,3))

            # state value
            Qprint = np.zeros([FLAGS.state_space])
            for i in range(FLAGS.state_space):
                ss = np.zeros([FLAGS.state_space]) # loop over one-hot encoding
                ss[i] = 1
                Qval = sess.run(QNet.Qout, feed_dict={QNet.state: ss.reshape(1, -1)}) # get Q-value of current state
                Qprint[i] = np.mean(Qval) # V = E[Q]
            print('V-Value: ')
            print(np.round(Qprint.reshape(3, 3), 3))
