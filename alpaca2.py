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
import matplotlib.pyplot as plt

from replay_buffer import replay_buffer
from square_environment import square_environment


# General Hyperparameters
tf.flags.DEFINE_integer("batch_size", 8, "Batch size for training")
tf.flags.DEFINE_integer("action_space", 5, "Dimensionality of action space")
tf.flags.DEFINE_integer("state_space", 26, "Dimensionality of state space")
tf.flags.DEFINE_integer("hidden_space", 32, "Dimensionality of hidden space")
tf.flags.DEFINE_float("gamma", 0.9, "Discount factor")
tf.flags.DEFINE_float("learning_rate", 2e-3, "Initial learning rate")
tf.flags.DEFINE_float("lr_drop", 1.0001, "Drop of learning rate per episodeI")
tf.flags.DEFINE_float("prior_precision", 0.25, "Prior precision (1/var)")
tf.flags.DEFINE_float("noise_precision", 0.01, "Noise precision (1/var)")
tf.flags.DEFINE_float("split", 0.5, "Split between fraction of trajectory used for updating posterior vs prior")
tf.flags.DEFINE_integer("N_episodes", 30000, "Number of episodes")
tf.flags.DEFINE_integer("N_tasks", 12, "Number of tasks")
tf.flags.DEFINE_integer("L_episode", 30, "Length of episodes")
tf.flags.DEFINE_integer("replay_memory_size", 1000, "Size of replay memory")
tf.flags.DEFINE_integer("update_freq", 5, "Update frequency of posterior and sampling of new policy")
tf.flags.DEFINE_integer("iter_amax", 3, "Number of iterations performed to determine amax")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)

class QNetwork():
    def __init__(self, scope="QNetwork"):
        self.gamma = FLAGS.gamma
        self.action_dim = FLAGS.action_space
        self.state_dim = FLAGS.state_space
        self.hidden_dim = FLAGS.hidden_space
        self.cprec = FLAGS.prior_precision
        self.lr = FLAGS.learning_rate
        self.iter_amax = FLAGS.iter_amax

        with tf.variable_scope(scope):
            # build graph
            self._build_model()

    def model(self, x):
        ''' Embedding into latent space '''
        with tf.variable_scope("latent", reuse=tf.AUTO_REUSE):
            # model architecture
            hidden1 = tf.contrib.layers.fully_connected(x, num_outputs=64, activation_fn=tf.nn.tanh)
            hidden2 = tf.contrib.layers.fully_connected(hidden1, num_outputs=self.hidden_dim* self.action_dim, activation_fn=tf.nn.tanh)
            hidden2_rs = tf.reshape(hidden2, [-1, self.hidden_dim, self.action_dim])

        return hidden2_rs

    def predict(self, phi):
        ''' predict Q-values of  next time step '''
        Wt = tf.reshape(self.wt, [1, self.hidden_dim, 1])
        W = tf.tile(Wt, [1, 1, self.action_dim])  # [input, latent space, action space, output space]
        output = tf.einsum('ijk,mjk->ik', phi, W)

        return output


    def _build_model(self):
        ''' constructing tensorflow model '''
        # placeholders ====================================================================
        # context data
        self.context_state = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name='state')  # input
        self.context_state_next = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name='next_state')  # input

        self.context_phi = self.model(self.context_state)  # latent space
        self.context_phi_next = self.model(self.context_state_next)  # latent space

        self.context_action = tf.placeholder(shape=[None], dtype=tf.int32, name='action')
        self.context_done = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='done')
        self.context_reward = tf.placeholder(shape=[None], dtype=tf.float32, name='reward')

        # for prediction
        self.state = tf.placeholder(shape=[None, self.state_dim], dtype = tf.float32, name='state') # input
        self.state_next = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name='next_state')  # input

        self.phi = self.model(self.state) # latent space
        self.phi_next = self.model(self.state_next)  # latent space

        self.action = tf.placeholder(shape=[None], dtype=tf.int32, name='action')
        self.done = tf.placeholder(shape=[None], dtype=tf.float32, name='done')
        self.reward = tf.placeholder(shape=[None], dtype=tf.float32, name='reward')

        self.learn = tf.placeholder(shape=[], dtype=tf.float32, name='learning_rate')
        self.nprec = tf.placeholder(shape=[], dtype=tf.float32, name= 'noise_precision')

        # output layer (Bayesian) =========================================================
        self.wt = tf.get_variable('wt', shape=[self.hidden_dim,1])
        self.Qout = tf.einsum('jm,bjk->bk', self.wt, self.phi, name='Qout')

        # prior (updated via GD) ---------------------------------------------------------
        self.w0_bar = tf.get_variable('w0_bar', dtype=tf.float32, shape=[self.hidden_dim,1])
        self.L0_asym = tf.get_variable('L0_asym', dtype=tf.float32, initializer=tf.sqrt(self.cprec)*tf.eye(self.hidden_dim)) # cholesky
        self.L0 = tf.matmul(self.L0_asym, tf.transpose(self.L0_asym))  # \Lambda_0

        self.sample_prior = self._sample_prior()

        (bs, _, _) = tf.unstack(tf.to_int32(tf.shape(self.context_phi)))
        self.Sigma_e_context = 1./self.nprec * tf.ones(bs, name='noise_precision')

        (bs, _, _) = tf.unstack(tf.to_int32(tf.shape(self.phi)))
        self.Sigma_e = 1./self.nprec * tf.ones(bs, name='noise_precision')

        self.Qout2 = tf.einsum('jm,bjk->bk', self.w0_bar, self.phi, name='Qout_0')

        # posterior (analytical update) --------------------------------------------------

        # phi(s, a)
        context_taken_action = tf.one_hot(tf.reshape(self.context_action, [-1, 1]), self.action_dim, dtype=tf.float32)
        context_phi_taken = tf.reduce_sum(tf.multiply(self.context_phi, context_taken_action), axis=2)

        taken_action = tf.one_hot(tf.reshape(self.action, [-1, 1]), self.action_dim, dtype=tf.float32)
        phi_taken = tf.reduce_sum(tf.multiply(self.phi, taken_action), axis=2)

        # update posterior if there is data
        (bs, _, _) = tf.unstack(tf.to_int32(tf.shape(self.context_phi)))
        self.wt_bar, self.Lt_inv = tf.cond(bs > 0,
                                            lambda: self._max_posterior(self.context_phi_next, context_phi_taken,
                                                                        self.context_reward),
                                            lambda: (self.w0_bar, tf.linalg.inv(self.L0)))

        self.Qout3 = tf.einsum('jm,bjk->bk', self.wt_bar, self.phi, name='Qout_t')

        self.Qout_lv = tf.einsum('jm,bjk->bk', self.wt_bar, self.phi) - \
                       (tf.einsum('bik,ij,bjk->bk', self.phi, self.Lt_inv, self.phi)+ \
                        tf.tile(tf.reshape(self.Sigma_e, [-1,1]), [1, self.action_dim]))

        self.Qout_uv = tf.einsum('jm,bjk->bk', self.wt_bar, self.phi) + \
                       (tf.einsum('bik,ij,bjk->bk', self.phi, self.Lt_inv, self.phi)+ \
                        tf.tile(tf.reshape(self.Sigma_e, [-1,1]), [1, self.action_dim]))

        # sample posterior
        with tf.control_dependencies([self.wt_bar, self.Lt_inv]):
            self.sample_post = self._sample_posterior(tf.reshape(self.wt_bar, [-1, 1]), self.Lt_inv)


        # current state -------------------------------------
        #taken_action = tf.one_hot(tf.reshape(self.action, [-1, 1]), self.action_dim, dtype=tf.float32)
        #phi_taken = tf.reduce_sum(tf.multiply(self.phi, taken_action), axis=2)
        self.Q = tf.einsum('jm,bj->b', self.wt_bar, phi_taken, name='Q')

        # next state ----------------------------------------
        Qnext = tf.einsum('jm,bjk->bk', self.wt_bar, self.phi_next, name='Qnext')

        max_action = tf.one_hot(tf.reshape(tf.argmax(Qnext, axis=1), [-1, 1]), self.action_dim, dtype=tf.float32)
        phi_max = tf.reduce_sum(tf.multiply(self.phi_next, max_action), axis=2)
        #phi_max = tf.stop_gradient(phi_max)

        # last factor to account for case that s is terminating state
        Qmax = tf.einsum('im,bi->b', self.wt_bar, phi_max, name='Qmax')
        self.Qtarget = self.reward + FLAGS.gamma * Qmax

        self.Qdiff = self.Qtarget - self.Q

        # combined ------------------------------------------
        self.phi_hat = phi_taken - self.gamma * phi_max
        self.phi_hat = tf.stop_gradient(self.phi_hat)
        Sigma_pred = tf.einsum('bi,ij,bj->b', self.phi_hat, self.Lt_inv, self.phi_hat, name='Sigma_pred')+  self.Sigma_e
        logdet_Sigma = tf.reduce_sum(tf.log(Sigma_pred))

        # loss function
        self.loss = tf.einsum('i,ik,k->', self.Qdiff, tf.linalg.inv(tf.linalg.diag(Sigma_pred)), self.Qdiff, name='loss')+ logdet_Sigma#tf.losses.mean_squared_error(self.Qtarget, self.Q)

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learn)

        ## Retrieve all trainable variables you defined in your graph
        tvs = tf.trainable_variables()
        ## Creation of a list of variables with the same shape as the trainable ones
        accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
        self.zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
        self.accum_loss = tf.Variable(0., 'loss')
        self.reset_loss = tf.assign(self.accum_loss, 0.)

        ## Calls the compute_gradients function of the optimizer to obtain... the list of gradients
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)

        ## Adds to each element from the list you initialized earlier with zeros its gradient (works because accum_vars and gvs are in the same order)
        self.accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(self.grads_and_vars) if gv[0] is not None]
        self.loss_ops = tf.assign(self.accum_loss, self.accum_loss+ self.loss)

        self.updateModel = self.optimizer.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(self.grads_and_vars) if gv[0] is not None])

        # summary
        loss_summary = tf.summary.scalar('loss', self.accum_loss)

        '''
        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []

        for g, v in accum_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("/grad/hist/%s" % v.name, g)
                grad_summaries.append(grad_hist_summary)
        '''

        cov_summary = tf.summary.image('covariance', tf.reshape(self.L0, [1, self.hidden_dim, self.hidden_dim, 1]))
        mu_summary = tf.summary.image('mean', tf.reshape(tf.tile(self.w0_bar, [1, self.hidden_dim]), [1, self.hidden_dim, self.hidden_dim, 1]))
        #wt_summary = tf.summary.image('wt', tf.reshape(tf.tile(self.wt, [1, self.hidden_dim]), [1, self.hidden_dim, self.hidden_dim, 1]))

        self.summaries_merged = tf.summary.merge([mu_summary, cov_summary, loss_summary])


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
        x = mu + tf.matmul(A, z)
        return x

    def _update_posterior(self, phi_hat, reward):
        ''' update posterior distribution '''
        Lt = tf.matmul(tf.transpose(phi_hat), tf.matmul(tf.linalg.inv(tf.linalg.diag(self.Sigma_e_context)), phi_hat)) + self.L0 # XXXXX
        Lt_inv = tf.matrix_inverse(Lt)
        wt_unnormalized = tf.matmul(self.L0, self.w0_bar) + \
                          tf.matmul(tf.transpose(phi_hat), tf.reshape(reward, [-1, 1]))
        wt_bar = tf.matmul(Lt_inv, wt_unnormalized)

        return wt_bar, Lt_inv

    def _max_posterior(self, phi_next, phi_taken, reward):
        ''' determine wt_bar for calculating phi(s_{t+1}, a*) '''
        # determine phi(max_action) based on Q determined by sampling wt from prior
        _ = self._sample_prior() # sample wt
        Q_next = tf.einsum('ijk,jl->ik', phi_next, self.w0_bar)
        max_action = tf.one_hot(tf.reshape(tf.argmax(Q_next, axis=1), [-1, 1]), self.action_dim, dtype=tf.float32)
        phi_max = tf.reduce_sum(tf.multiply(phi_next, max_action), axis=2)

        # iterations
        for _ in range(self.iter_amax):
            # sample posterior
            phi_hat = phi_taken - self.gamma* phi_max

            # update posterior distributior
            wt_bar, Lt_inv = self._update_posterior(phi_hat, reward)

            # determine phi(max_action) based on Q determined by sampling wt from posterior
            _ = self._sample_posterior(wt_bar, Lt_inv)
            #Q_next = self.predict((phi_next))
            Q_next = tf.einsum('i,jik->jk', tf.reshape(self.wt, [-1]), phi_next)

            max_action = tf.one_hot(tf.reshape(tf.argmax(Q_next, axis=1), [-1, 1]), self.action_dim, dtype=tf.float32)
            phi_max = tf.reduce_sum(tf.multiply(phi_next, max_action), axis=2)

        # stop gradient
        phi_hat = tf.stop_gradient(phi_hat)
        # update posterior distribution
        wt_bar, Lt_inv = self._update_posterior(phi_hat, reward)

        return wt_bar, Lt_inv

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
    if np.random.rand() >= epsilon:
        action = np.argmax(x)
    else:
        action = np.random.randint(FLAGS.action_space)

    return action


def plot_V(path, target, trajectory, V):
    fig, ax = plt.subplots()
    im = ax.imshow(V.reshape(5, 5))

    pos = np.argmax(np.array(trajectory), axis=1)
    # i/5 downwards, i%5 rightwards.
    xpos = pos % 5
    ypos = pos / 5

    ax.plot(target[1], target[0], 'ro', markersize=20)
    ax.plot(xpos, ypos, 'bo', markersize=20)

    fig.colorbar(im)
    plt.savefig(path)
    plt.close()


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

# folder for plotting
V_0_dir = 'Figures/'+ time.strftime('%H-%M-%d_%m-%y')+ '/Q_Fcn_0/'
if not os.path.exists(V_0_dir):
    os.makedirs(V_0_dir)

V_t_dir = 'Figures/'+ time.strftime('%H-%M-%d_%m-%y')+ '/Q_Fcn_t/'
if not os.path.exists(V_t_dir):
    os.makedirs(V_t_dir)

V_TS_dir = 'Figures/'+ time.strftime('%H-%M-%d_%m-%y')+ '/Q_TS_Fcn/'
if not os.path.exists(V_TS_dir):
    os.makedirs(V_TS_dir)

Lt_inv_dir = 'Figures/'+ time.strftime('%H-%M-%d_%m-%y')+ '/Lt_inv/'
if not os.path.exists(Lt_inv_dir):
    os.makedirs(Lt_inv_dir)

phi_dir = 'Figures/' + time.strftime('%H-%M-%d_%m-%y') + '/phi/'
if not os.path.exists(phi_dir):
    os.makedirs(phi_dir)

# initialize replay memory and model
fullbuffer = replay_buffer(FLAGS.replay_memory_size) # large buffer to store all experience
tempbuffer = replay_buffer(FLAGS.L_episode) # buffer for episode
log.info('Build Tensorflow Graph')
QNet = QNetwork() # neural network

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
    learning_rate = FLAGS.learning_rate
    noise_precision = FLAGS.noise_precision
    step_count = [] # count steps required to accomplish task
    state_count = [] # count number of visitations of each state
    reward_count = []

    # -----------------------------------------------------------------------------------
    # loop episodes
    print("Episodes...")
    log.info('Loop over episodes')
    for episode in range(FLAGS.N_episodes):

        # count state visitations
        st = []

        # count reward
        rw = []

        # target positions
        target = []

        # loop tasks --------------------------------------------------------------------
        for n in range(FLAGS.N_tasks):
            # initialize buffer
            tempbuffer.reset()
            # reset environment
            state = env.reset()

            st.append(state[:-1])
            rw.append(0)
            target.append(env.target)

            # sample w from prior
            sess.run([QNet.sample_prior])

            # loop steps
            step = 0

            while step < FLAGS.L_episode:
                # take a step
                Qval = sess.run([QNet.Qout], feed_dict={QNet.state: state.reshape(1,-1)})
                action = eGreedyAction(Qval, eps)
                next_state, reward, done = env.step(action)

                # store experience in memory
                new_experience = [state, action, reward, next_state, done]
                tempbuffer.add(new_experience)

                # update posterior
                # TODO: could speed up by iteratively adding
                if step % FLAGS.update_freq == 0 and step != 0:
                    reward_train = np.zeros([step+1, ])
                    state_train = np.zeros([step+1, FLAGS.state_space])
                    next_state_train = np.zeros([step+1, FLAGS.state_space])
                    action_train = np.zeros([step+1, ])
                    done_train = np.zeros([step+1, 1])

                    for k, experience in enumerate(tempbuffer.buffer):
                        # [s, a, r, s', a*, d]
                        state_train[k] = experience[0]
                        action_train[k] = experience[1]
                        reward_train[k] = experience[2]
                        next_state_train[k] = experience[3]
                        done_train[k] = experience[4]

                    sess.run(QNet.sample_post,
                             feed_dict={QNet.context_state: state_train, QNet.context_action: action_train,
                                        QNet.context_reward: reward_train, QNet.context_state_next: next_state_train,
                                        QNet.context_done: done_train,
                                        QNet.nprec: noise_precision})

                    # State Value Fcn -----------------------------------------------------------
                    if (episode) % 1000 == 0 and n == 0:
                        tar = env.target  # target location as in array notation i.e. tar[0] downwards, tar[1] rightwards

                        ss = np.zeros([FLAGS.state_space - 1, FLAGS.state_space])  # loop over one-hot encoding
                        for i in range(FLAGS.state_space - 1):
                            ss[i,i] = 1
                            # i/5 downwards, i%5 rightwards
                            if i / 5 == tar[0] and i % 5 == tar[1]:  # add reward at target location
                                ss[i,FLAGS.state_space - 1] = 0.

                        _, Qout0, Qoutt, Qout_lv, Qout_uv, Lt_inv = sess.run([QNet.sample_post, QNet.Qout2, QNet.Qout3, QNet.Qout_lv, QNet.Qout_uv, QNet.Lt_inv],
                                                feed_dict={QNet.state: ss.reshape(FLAGS.state_space - 1, -1),
                                                QNet.context_state: state_train, QNet.context_action: action_train,
                                                QNet.context_reward: reward_train,
                                                QNet.context_state_next: next_state_train,
                                                QNet.context_done: done_train,
                                                QNet.nprec: noise_precision})

                        V_0 = np.max(Qout0, axis=1)
                        V_t = np.max(Qoutt, axis=1)
                        V_lv = np.max(Qout_lv, axis=1)
                        V_uv = np.max(Qout_uv, axis=1)


                        plot_V(V_0_dir+'Epoch_'+str(episode)+'_Step_'+str(step), tar, st, V_0)

                        plot_V(V_t_dir+'Epoch_'+str(episode)+'_Step_'+str(step), tar, st, V_t)

                        plot_V(V_t_dir+'Epoch_'+str(episode)+'_Step_'+str(step)+'_LB', tar, st, V_lv)

                        plot_V(V_t_dir+'Epoch_'+str(episode)+'_Step_'+str(step)+'_UB', tar, st, V_uv)

                        fig, ax = plt.subplots()
                        im = ax.imshow(Lt_inv)
                        fig.colorbar(im)
                        plt.savefig(Lt_inv_dir+'Epoch_'+str(episode)+'_Step_'+str(step))
                        plt.close()

                    # -----------------------------------------------------------------------

                # update state, and counters
                state = next_state
                global_index += 1
                step += 1

                # count state visitations
                st.append(state[:-1])

                # count rewad
                rw.append(reward)

                # target positions
                target.append(env.target)

                # State-Value Fcn -----------------------------------------------------------
                if (episode) % 1000 == 0 and n == 0:
                    tar = env.target # target location as in array notation i.e. tar[0] downwards, tar[1] rightwards
                    # state value
                    V_TS = np.zeros([FLAGS.state_space - 1])
                    for i in range(FLAGS.state_space - 1):
                        ss = np.zeros([FLAGS.state_space])  # loop over one-hot encoding
                        ss[i] = 1

                        Qout, phi = sess.run([QNet.Qout, QNet.phi], feed_dict={QNet.state: ss.reshape(1, -1)})
                        V_TS[i] = np.max(Qout)

                        # i/5 downwards, i%5 rightwards
                        if (i / 5 == 0 or i/5 == 4) and i % 5 == 4:  # add reward at target location
                            # if i / 5 == tar[0] and i % 5 == tar[1]:
                            ss[FLAGS.state_space - 1] = 1.

                        phir = sess.run([QNet.phi], feed_dict={QNet.state: ss.reshape(1, -1)})


                        if step == 1:
                            fig, ax = plt.subplots(ncols = 2)
                            ax[0].imshow(phi[0])
                            im = ax[1].imshow(phir[0][0])
                            fig.colorbar(im)
                            plt.savefig(phi_dir+'Epoch_'+str(episode)+'_State_'+str(i))
                            plt.close()

                    #
                    plot_V(V_TS_dir+'Epoch_'+str(episode)+'_Step_'+str(step), tar, state.reshape(1,-1), V_TS)

                # -----------------------------------------------------------------------

                # check if s is final state
                # if done:
                #    break

            '''
            if episode % 100 == 0:
                fig.colorbar(im)
                plt.savefig('Figures/Epoch_'+ str(episode)+ '_Task_'+ str(n))
                plt.close()
            '''

            # append episode buffer to large buffer
            fullbuffer.add(tempbuffer.buffer)


        # epsilon-greedy schedule
        #eps-= deps

        log.info('Episode %3.d with R %3.d', episode, np.sum(rw))

        learning_rate /= FLAGS.lr_drop

        if episode < 9000:
            noise_precision*= (1.+ 5e-4)
        else:
            noise_precision += 5e-4

        # sample from larger buffer [s, a, r, s', d] with current experience not yet included
        experience = fullbuffer.sample(FLAGS.batch_size)

        reward_count.append(np.sum(np.array(rw)))

        # reset gradients
        sess.run([QNet.zero_ops, QNet.reset_loss])

        for e in range(FLAGS.batch_size):

            state_sample = np.zeros((FLAGS.L_episode, FLAGS.state_space))
            action_sample = np.zeros((FLAGS.L_episode,))
            reward_sample = np.zeros((FLAGS.L_episode,))
            next_state_sample = np.zeros((FLAGS.L_episode, FLAGS.state_space))
            done_sample = np.zeros((FLAGS.L_episode,))

            for k, (s0, a, r, s1, d) in enumerate(experience[e]):
                state_sample[k] = s0
                action_sample[k] = a
                reward_sample[k] = r
                next_state_sample[k] = s1
                done_sample[k] = d

            # split in train and validation set
            #train = np.random.choice(np.arange(FLAGS.L_episode), np.int(FLAGS.split * FLAGS.L_episode), replace=False)  # mixed
            #valid = np.setdiff1d(np.arange(FLAGS.L_episode), train)

            split = np.int(FLAGS.L_episode* np.random.rand())

            train = np.arange(0, split)
            valid = np.arange(split, FLAGS.L_episode)

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

            _, _, w0_bar, L0_bar, wt = sess.run([QNet.accum_ops, QNet.loss_ops, QNet.w0_bar, QNet.L0, QNet.wt],
                                                feed_dict={QNet.context_state: state_train, QNet.context_action: action_train,
                                                           QNet.context_reward: reward_train, QNet.context_state_next: next_state_train,
                                                           QNet.state: state_valid, QNet.action: action_valid,
                                                           QNet.reward: reward_valid, QNet.state_next: next_state_valid,
                                                           QNet.learn: learning_rate,
                                                           QNet.nprec: noise_precision})


        # update summary
        _, summaries_merged = sess.run([QNet.updateModel, QNet.summaries_merged], feed_dict={QNet.learn: learning_rate})

        reward_summary = tf.Summary(value=[tf.Summary.Value(tag='Episodic Reward', simple_value=np.sum(np.array(rw)))])

        summary_writer.add_summary(reward_summary, global_index)
        summary_writer.add_summary(summaries_merged, global_index)
        summary_writer.flush()

        # ================================================================
        # print to console
        if episode % 1000 == 0:
            print('Reward in Episode ' + str(episode)+  ':   '+ str(np.sum(rw)))
            print('Learning_rate: '+ str(learning_rate)+ ', Nprec: '+ str(noise_precision))
            print('State Count: ')
            state_count = np.sum(np.array(st), axis=0)
            print(state_count.reshape(5,5))

            #print(np.round(wt.reshape(1,-1), 3))
            np.set_printoptions(suppress=True)
            #print(np.round(L0_bar.reshape(FLAGS.hidden_space, FLAGS.hidden_space), 3))


    plt.figure()
    plt.plot(reward_count)
    plt.show()

    fullbuffer.reset()
    tempbuffer.reset()

    # update model and prior

# log file
# parser file
# TF summaries
# save model to restore