'''
Class structure for ALPaCA meta-RL code
'''

import numpy as np
import tensorflow as tf
import os
import time
import logging
import matplotlib.pyplot as plt

from mountain_car import MountainCarEnv

import sys
sys.path.insert(0, './..')
from replay_buffer import replay_buffer

np.random.seed(1234)

# General Hyperparameters
tf.flags.DEFINE_integer("batch_size", 2, "Batch size for training")
tf.flags.DEFINE_integer("action_space", 3, "Dimensionality of action space")
tf.flags.DEFINE_integer("state_space", 2, "Dimensionality of state space")
tf.flags.DEFINE_integer("hidden_space", 32, "Dimensionality of hidden space") # change
tf.flags.DEFINE_integer("latent_space", 16, "Dimensionality of hidden space") # change
tf.flags.DEFINE_float("gamma", 0.95, "Discount factor")
tf.flags.DEFINE_float("learning_rate", 1e-2, "Initial learning rate") # change
tf.flags.DEFINE_float("lr_drop", 1.00015, "Drop of learning rate per episode") # change
tf.flags.DEFINE_float("prior_precision", 0.1, "Prior precision (1/var)")
tf.flags.DEFINE_float("noise_precision", 0.2, "Noise precision (1/var)") # change
tf.flags.DEFINE_integer("N_episodes", 30000, "Number of episodes")
tf.flags.DEFINE_integer("N_tasks", 2, "Number of tasks")
tf.flags.DEFINE_integer("L_episode", 200, "Length of episodes")
tf.flags.DEFINE_integer("replay_memory_size", 50, "Size of replay memory")
tf.flags.DEFINE_integer("update_freq", 220 , "Update frequency of posterior and sampling of new policy")
tf.flags.DEFINE_integer("iter_amax", 3, "Number of iterations performed to determine amax")
tf.flags.DEFINE_string('non_linearity', 'tanh', 'Non-linearity used in encoder.')

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)

class QNetwork():
    def __init__(self, scope="QNetwork"):
        self.gamma = FLAGS.gamma
        self.action_dim = FLAGS.action_space
        self.state_dim = FLAGS.state_space
        self.hidden_dim = FLAGS.hidden_space
        self.latent_dim = FLAGS.latent_space
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
            hidden1 = tf.contrib.layers.fully_connected(x, num_outputs=self.hidden_dim, activation_fn=tf.nn.relu)
            hidden2 = tf.contrib.layers.fully_connected(hidden1, num_outputs=self.hidden_dim, activation_fn=tf.nn.relu)
            hidden3 = tf.contrib.layers.fully_connected(hidden2, num_outputs=self.latent_dim, activation_fn=tf.nn.relu)

            # bring it into the right order of shape [batch_size, hidden_dim, action_dim]
            # needs to be done this manner due to the way tf reshapes arrays
            hidden3_rs = tf.reshape(hidden3, [-1, self.action_dim, self.latent_dim])
            hidden3_rs = tf.transpose(hidden3_rs, [0, 2, 1])


        return hidden3_rs

    def state_trafo(self, state, action):
        ''' append action to the state '''
        state = tf.expand_dims(state, axis=1)
        state = tf.tile(state, [1, 1, self.action_dim])
        state = tf.reshape(state, [-1, self.state_dim])

        #action = tf.one_hot(action, self.action_dim, dtype=tf.float32)
        #state = tf.concat([state, action], axis = 1)
        state = tf.concat([state, tf.reshape(action, [-1,1])], axis = 1)

        return state


    def _build_model(self):
        ''' constructing tensorflow model '''
        #
        self.lr_placeholder = tf.placeholder(shape=[], dtype=tf.float32, name='learning_rate')

        # placeholders ====================================================================
        ## context data
        self.context_state = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name='state')  # input
        self.context_state_next = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name='next_state')  # input

        # append action s.t. it is an input to the network
        (bsc, _) = tf.unstack(tf.to_int32(tf.shape(self.context_state)))

        context_action_augm = tf.range(self.action_dim, dtype=tf.int32)
        context_action_augm = tf.tile(context_action_augm, [bsc])

        context_state = self.state_trafo(self.context_state, context_action_augm)
        context_state_next = self.state_trafo(self.context_state_next, context_action_augm)

        # latent representation
        self.context_phi = self.model(context_state)  # latent space
        self.context_phi_next = self.model(context_state_next)  # latent space

        self.context_action = tf.placeholder(shape=[None], dtype=tf.int32, name='action')
        self.context_done = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='done')
        self.context_reward = tf.placeholder(shape=[None], dtype=tf.float32, name='reward')

        ## prediction data
        self.state = tf.placeholder(shape=[None, self.state_dim], dtype = tf.float32, name='state') # input
        self.state_next = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name='next_state')  # input

        # append action s.t. it is an input to the network
        (bs, _) = tf.unstack(tf.to_int32(tf.shape(self.state)))

        action_augm = tf.range(self.action_dim, dtype=tf.int32)
        action_augm = tf.tile(action_augm, [bs])

        state = self.state_trafo(self.state, action_augm)
        state_next = self.state_trafo(self.state_next, action_augm)

        # latent representation
        self.phi = self.model(state) # latent space
        self.phi_next = self.model(state_next)  # latent space

        self.action = tf.placeholder(shape=[None], dtype=tf.int32, name='action')
        self.done = tf.placeholder(shape=[None], dtype=tf.float32, name='done')
        self.reward = tf.placeholder(shape=[None], dtype=tf.float32, name='reward')

        # noise variance
        self.nprec = tf.placeholder(shape=[], dtype=tf.float32, name= 'noise_precision')

        self.Sigma_e_context = 1. / self.nprec * tf.ones(bsc, name='noise_precision')
        self.Sigma_e = 1. / self.nprec * tf.ones(bs, name='noise_precision')

        # output layer (Bayesian) =========================================================
        self.wt = tf.get_variable('wt', shape=[self.latent_dim,1], trainable=False)
        self.Qout = tf.einsum('jm,bjk->bk', self.wt, self.phi, name='Qout')

        # prior (updated via GD) ---------------------------------------------------------
        self.w0_bar = tf.get_variable('w0_bar', dtype=tf.float32, shape=[self.latent_dim,1])
        self.L0_asym = tf.get_variable('L0_asym', dtype=tf.float32, initializer=tf.sqrt(self.cprec)*tf.eye(self.latent_dim)) # cholesky
        self.L0 = tf.matmul(self.L0_asym, tf.transpose(self.L0_asym))  # \Lambda_0

        self.sample_prior = self._sample_prior()

        # posterior (analytical update) --------------------------------------------------
        # phi(s, a)
        context_taken_action = tf.one_hot(tf.reshape(self.context_action, [-1, 1]), self.action_dim, dtype=tf.float32)
        self.context_phi_taken = tf.reduce_sum(tf.multiply(self.context_phi, context_taken_action), axis=2)

        taken_action = tf.one_hot(tf.reshape(self.action, [-1, 1]), self.action_dim, dtype=tf.float32)
        phi_taken = tf.reduce_sum(tf.multiply(self.phi, taken_action), axis=2)

        # update posterior if there is data
        self.wt_bar, self.Lt_inv = tf.cond(bsc > 0,
                                            lambda: self._max_posterior(self.context_phi_next, self.context_phi_taken,
                                                                        self.context_reward),
                                            lambda: (self.w0_bar, tf.linalg.inv(self.L0)))

        # sample posterior
        with tf.control_dependencies([self.wt_bar, self.Lt_inv]):
            self.sample_post = self._sample_posterior(tf.reshape(self.wt_bar, [-1, 1]), self.Lt_inv)

        # loss function ==================================================================
        # current state -------------------------------------
        self.Q = tf.einsum('im,bi->b', self.wt_bar, phi_taken, name='Q')

        # next state ----------------------------------------
        Qnext = tf.einsum('jm,bjk->bk', self.wt_bar, self.phi_next, name='Qnext')

        max_action = tf.one_hot(tf.reshape(tf.argmax(Qnext, axis=1), [-1, 1]), self.action_dim, dtype=tf.float32)
        phi_max = tf.reduce_sum(tf.multiply(self.phi_next, max_action), axis=2)
        phi_max = tf.stop_gradient(phi_max)

        # last factor to account for case that s is terminating state
        Qmax = tf.einsum('im,bi->b', self.wt_bar, phi_max, name='Qmax')
        self.Qtarget = self.reward + FLAGS.gamma * tf.multiply(1- self.done, Qmax)
        #self.Qtarget = tf.stop_gradient(self.Qtarget)

        # Q(s',a*)+ r- Q(s,a)
        self.Qdiff = self.Qtarget - self.Q

        # phi_hat* Lt_inv* phi_hat --------------------------
        self.phi_hat = phi_taken - self.gamma * phi_max
        #self.phi_hat = tf.stop_gradient(self.phi_hat)
        Sigma_pred = tf.einsum('bi,ij,bj->b', self.phi_hat, self.Lt_inv, self.phi_hat, name='Sigma_pred')+  self.Sigma_e
        logdet_Sigma = tf.reduce_sum(tf.log(Sigma_pred))

        # loss
        self.loss0 = tf.einsum('i,i->', self.Qdiff, self.Qdiff,name='loss0')
        self.loss1 = tf.einsum('i,ik,k->', self.Qdiff, tf.linalg.inv(tf.linalg.diag(Sigma_pred)), self.Qdiff, name='loss')
        self.loss2 = logdet_Sigma
        self.loss = self.loss1+ self.loss2 #tf.losses.mean_squared_error(self.Qtarget, self.Q)

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_placeholder)

        # gradient
        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.gradient_holders.append(placeholder)

        # symbolic gradient of loss w.r.t. tvars
        self.gradients = self.optimizer.compute_gradients(self.loss, tvars)

        #
        self.updateModel = self.optimizer.apply_gradients(zip(self.gradient_holders, tvars))

        # summaries ========================================================================
        variables_names = [v.name for v in tf.trainable_variables()]

        # Keep track of gradient values
        grad_summaries = []
        for idx, var in zip(variables_names, self.gradient_holders):
            grad_hist_summary = tf.summary.histogram("/grad/hist/%s" % idx, var)
            grad_summaries.append(grad_hist_summary)

        # keep track of weights
        tvars = tf.trainable_variables()
        weight_summary = []
        for idx, var in zip(variables_names, tvars):
            weight_hist_summary = tf.summary.histogram("/weight/hist/%s" % idx, var)
            weight_summary.append(weight_hist_summary)

        # prior last layer summaries
        cov_summary = tf.summary.image('covariance', tf.reshape(self.L0, [1, self.latent_dim, self.latent_dim, 1]))
        mu_summary = tf.summary.image('mean', tf.reshape(tf.tile(self.w0_bar, [1, self.latent_dim]), [1, self.latent_dim, self.latent_dim, 1]))

        # concat summaries
        self.summaries_merged = tf.summary.merge([mu_summary, cov_summary, grad_summaries, weight_summary])


    def _sample_prior(self):
        ''' sample wt from prior '''
        update_op = tf.assign(self.wt, self._sample_MN(self.w0_bar, tf.matrix_inverse(self.L0)))
        return update_op

    def _sample_posterior(self, wt_bar, Lt_inv):
        ''' sample wt from posterior '''
        update_op = tf.assign(self.wt, self._sample_MN(wt_bar, Lt_inv))
        return update_op

    def _sample_MN(self, mu, cov):
        ''' sample from multi-variate normal '''
        #A = tf.linalg.cholesky(cov)
        V, U = tf.linalg.eigh(cov)
        z = tf.random_normal(shape=[self.latent_dim,1])
        #x = mu + tf.matmul(A, z)
        x = mu+ tf.matmul(tf.matmul(U, tf.sqrt(tf.linalg.diag(V))), z)
        return x

    def _update_posterior(self, phi_hat, reward):
        ''' update posterior distribution '''
        # I've done that differently than outlined in the write-up
        # since I don't like to put the noise variance inside the prior
        Le = tf.linalg.inv(tf.linalg.diag(self.Sigma_e_context)) # noise precision
        Lt = tf.matmul(tf.transpose(phi_hat), tf.matmul(Le, phi_hat)) + self.L0 # posterior precision
        Lt_inv = tf.matrix_inverse(Lt) # posterior variance
        wt_unnormalized = tf.matmul(self.L0, self.w0_bar) + \
                          tf.matmul(tf.transpose(phi_hat), tf.matmul(Le, tf.reshape(reward, [-1, 1])))
        wt_bar = tf.matmul(Lt_inv, wt_unnormalized) # posterior mean

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
            Q_next = tf.einsum('i,jik->jk', tf.reshape(self.wt, [-1]), phi_next)

            max_action = tf.one_hot(tf.reshape(tf.argmax(Q_next, axis=1), [-1, 1]), self.action_dim, dtype=tf.float32)
            phi_max = tf.reduce_sum(tf.multiply(phi_next, max_action), axis=2)

        # stop gradient through context
        phi_hat = tf.stop_gradient(phi_hat)

        # update posterior distribution
        wt_bar, Lt_inv = self._update_posterior(phi_hat, reward)

        return wt_bar, Lt_inv


def eGreedyAction(x, epsilon=0.9):
    ''' select next action according to epsilon-greedy algorithm '''
    if np.random.rand() >= epsilon:
        action = np.argmax(x)
    else:
        action = np.random.randint(FLAGS.action_space)

    return action


# Main Routine ===========================================================================
#
batch_size = FLAGS.batch_size
eps = 0.

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

fh = logging.FileHandler(logger_dir+'tensorflow_'+ time.strftime('%H-%M-%d_%m-%y')+ '.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)

# write hyperparameters to logger
log.info('Parameters')
for key in FLAGS.__flags.keys():
    log.info('{}={}'.format(key, getattr(FLAGS, key)))

# folder to save and restore model -------------------------------------------------------
saver_dir = './model/'+ time.strftime('%H-%M-%d_%m-%y')+ '/'
if not os.path.exists(saver_dir):
    os.makedirs(saver_dir)

# folder for plotting --------------------------------------------------------------------
Vt_dir = 'figures/'+ time.strftime('%H-%M-%d_%m-%y')+ '/rt/'
if not os.path.exists(Vt_dir):
    os.makedirs(Vt_dir)

V0_dir = 'figures/'+ time.strftime('%H-%M-%d_%m-%y')+ '/r0/'
if not os.path.exists(V0_dir):
    os.makedirs(V0_dir)

# initialize replay memory and model
fullbuffer = replay_buffer(FLAGS.replay_memory_size) # large buffer to store all experience
tempbuffer = replay_buffer(FLAGS.L_episode) # buffer for episode
log.info('Build Tensorflow Graph')
QNet = QNetwork() # neural network

# initialize environment
env = MountainCarEnv() #gym.make('MountainCar-v0')

with tf.Session() as sess:
    # set random seed
    tf.set_random_seed(1234)

    # session
    init = tf.global_variables_initializer()
    sess.run(init)

    # checkpoint and summaries
    log.info('Save model snapshot')
    saver = tf.train.Saver(max_to_keep=4)
    filename = os.path.join(saver_dir, 'model')
    saver.save(sess, filename)

    summary_writer = tf.summary.FileWriter(logdir=os.path.join('./', 'summaries/', time.strftime('%H-%M-%d_%m-%y')), graph=sess.graph)

    # initialize
    global_index = 0 # counter
    learning_rate = FLAGS.learning_rate
    noise_precision = FLAGS.noise_precision

    gradBuffer = sess.run(tf.trainable_variables()) # get shapes of tensors

    for idx in range(len(gradBuffer)):
        gradBuffer[idx] *= 0


    # loss buffers to visualize in tensorboard
    lossBuffer = 0
    loss0Buffer = 0
    loss1Buffer = 0
    loss2Buffer = 0

    # -----------------------------------------------------------------------------------
    # loop episodes
    print("Episodes...")
    log.info('Loop over episodes')
    for episode in range(FLAGS.N_episodes):

        # time measurement
        start = time.time()

        # count reward
        rw = []

        # loop tasks --------------------------------------------------------------------
        for n in range(FLAGS.N_tasks):
            # initialize buffer
            tempbuffer.reset()

            # reset state
            state = env.reset()

            rw.append(0)

            # sample w from prior
            sess.run([QNet.sample_prior])

            # loop steps
            step = 0

            while step < FLAGS.L_episode:

                # take a step
                Qval = sess.run([QNet.Qout], feed_dict={QNet.state: state.reshape(-1,FLAGS.state_space)})
                action = eGreedyAction(Qval, eps)
                next_state, reward, done, _ = env.step(action)

                # store experience in memory
                new_experience = [state, action, reward, next_state, done]
                tempbuffer.add(new_experience)

                # count rewad
                rw.append(reward)

                # update state, and counters
                state = next_state
                global_index += 1
                step += 1

                if done == 1:
                    break

                # update posterior
                # TODO: could speed up by iteratively adding
                if step % FLAGS.update_freq == 0 and step != 0:
                    reward_train = np.zeros([step+1, ])
                    state_train = np.zeros([step+1, FLAGS.state_space])
                    next_state_train = np.zeros([step+1, FLAGS.state_space])
                    action_train = np.zeros([step+1, ])
                    done_train = np.zeros([step+1, 1])

                    # fill arrays
                    for k, experience in enumerate(tempbuffer.buffer):
                        # [s, a, r, s', a*, d]
                        state_train[k] = experience[0]
                        action_train[k] = experience[1]
                        reward_train[k] = experience[2]
                        next_state_train[k] = experience[3]
                        done_train[k] = experience[4]

                    # update
                    _, wt_bar, Lt_inv = sess.run([QNet.sample_post, QNet.wt_bar, QNet.Lt_inv],
                             feed_dict={QNet.context_state: state_train, QNet.context_action: action_train,
                                        QNet.context_reward: reward_train, QNet.context_state_next: next_state_train,
                                        QNet.context_done: done_train,
                                        QNet.nprec: noise_precision})

                    # plot Value function
                    if episode % 1000 == 0 and n == 0:
                        # generate grid
                        minp = env.min_position
                        maxp = env.max_position
                        maxs = env.max_speed

                        velgrid = np.linspace(-maxs, maxs, 10)
                        posgrid = np.linspace(-minp, maxp, 10)

                        mesh = np.meshgrid(posgrid, velgrid)
                        meshgrid = np.concatenate([mesh[0].reshape(-1,1), mesh[1].reshape(-1,1)], axis=1)

                        # value function
                        phi_mesh= sess.run([QNet.phi], feed_dict={QNet.state: meshgrid})

                        print(phi_mesh.shape)

                        Qmesh = np.dot(wt_bar, phi_mesh)

                        Vmesh = np.max(Qmesh, axis=1)

                        fig, ax = plt.subplots(ncols=1, figsize=[15, 5])
                        ax.imshow(Qmesh.reshape(10, 10))
                        plt.savefig(Vt_dir + 'Epoch_' + str(episode) + '_step_' + str(step) + '_Reward')
                        plt.close()

                # -----------------------------------------------------------------------

            # append episode buffer to large buffer
            fullbuffer.add(tempbuffer.buffer)

        if episode % 1000 == 0:
            log.info('Episode %3.d with R %3.d', episode, np.sum(rw))

        # learning rate schedule
        if learning_rate > 2e-4:
            learning_rate /= FLAGS.lr_drop

        if noise_precision <2 and episode% 1500 == 0:
            noise_precision*= 1.5


        # Gradient descent
        for e in range(batch_size):

            # sample from larger buffer [s, a, r, s', d] with current experience not yet included
            experience = fullbuffer.sample(1)

            L_experience = len(experience[0])

            state_sample = np.zeros((L_experience, FLAGS.state_space))
            action_sample = np.zeros((L_experience,))
            reward_sample = np.zeros((L_experience,))
            next_state_sample = np.zeros((L_experience, FLAGS.state_space))
            done_sample = np.zeros((L_experience,))

            # fill arrays
            for k, (s0, a, r, s1, d) in enumerate(experience[0]):
                state_sample[k] = s0
                action_sample[k] = a
                reward_sample[k] = r
                next_state_sample[k] = s1
                done_sample[k] = d

            # split into context and prediction set
            split = 0 #np.int(0.7* FLAGS.L_episode* np.random.rand())

            train = np.arange(0, split)
            valid = np.arange(split, L_experience)

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

            # update model
            grads, loss0, loss1, loss2, loss = sess.run([QNet.gradients, QNet.loss0, QNet.loss1, QNet.loss2, QNet.loss],
                                                feed_dict={QNet.context_state: state_train, QNet.context_action: action_train,
                                                           QNet.context_reward: reward_train, QNet.context_state_next: next_state_train,
                                                           QNet.state: state_valid, QNet.action: action_valid,
                                                           QNet.reward: reward_valid, QNet.state_next: next_state_valid,
                                                           QNet.done: done_valid,
                                                           QNet.lr_placeholder: learning_rate,
                                                           QNet.nprec: noise_precision})


            for idx, grad in enumerate(grads): # grad[0] is gradient and grad[1] the variable itself
                gradBuffer[idx] += (grad[0]/ batch_size)

            lossBuffer += loss
            loss0Buffer += loss0
            loss1Buffer += loss1
            loss2Buffer += loss2

        # update summary
        feed_dict= dictionary = dict(zip(QNet.gradient_holders, gradBuffer))
        feed_dict.update({QNet.lr_placeholder: learning_rate})
        _, summaries_merged = sess.run([QNet.updateModel, QNet.summaries_merged], feed_dict=feed_dict)

        loss_summary = tf.Summary(value=[tf.Summary.Value(tag='Loss', simple_value=(lossBuffer / batch_size))])
        loss0_summary = tf.Summary(value=[tf.Summary.Value(tag='TD_Loss', simple_value=(loss0Buffer / batch_size))])
        loss1_summary = tf.Summary(value=[tf.Summary.Value(tag='TDW_Loss', simple_value=(loss1Buffer/ batch_size))])
        loss2_summary = tf.Summary(value=[tf.Summary.Value(tag='Sig_Loss', simple_value=(loss2Buffer / batch_size))])
        reward_summary = tf.Summary(value=[tf.Summary.Value(tag='Episodic Reward', simple_value=np.sum(np.array(rw)))])
        #variance_summary = tf.Summary(value=[tf.Summary.Value(tag='Position Variance', simple_value=(np.max(state_sample[:,0])- np.min(state_sample[:,0])))])

        learning_rate_summary = tf.Summary(value=[tf.Summary.Value(tag='Learning rate', simple_value=learning_rate)])

        summary_writer.add_summary(loss_summary, episode)
        summary_writer.add_summary(loss0_summary, episode)
        summary_writer.add_summary(loss1_summary, episode)
        summary_writer.add_summary(loss2_summary, episode)
        summary_writer.add_summary(reward_summary, episode)
        #summary_writer.add_summary(variance_summary, episode)
        summary_writer.add_summary(summaries_merged, episode)
        summary_writer.add_summary(learning_rate_summary, episode)

        # reset buffers
        for idx in range(len(gradBuffer)):
            gradBuffer[idx] *= 0

        summary_writer.flush()

        lossBuffer *= 0
        loss0Buffer *= 0
        loss1Buffer *= 0
        loss2Buffer *= 0

        # increase the batch size after the first episode. Would allow N_tasks < batch_size due to buffer
        if episode < 2:
            batch_size *= 2

        # ===============================================================
        # save model

        #if episode > 0 and episode % 1000 == 0:
          # Save a checkpoint
          #log.info('Save model snapshot')

          #filename = os.path.join(saver_dir, 'model')
          #saver.save(sess, filename, global_step=episode, write_meta_graph=False)

        # ================================================================
        # print to console
        if episode % 500 == 0:
            print('Reward in Episode ' + str(episode)+  ':   '+ str(np.sum(rw)))
            print('Learning_rate: '+ str(np.round(learning_rate,5))+ ', Nprec: '+ str(noise_precision))

            # plot Value function
            Npts = 20
            # generate grid
            minp = env.min_position
            maxp = env.max_position
            maxs = env.max_speed

            velgrid = np.linspace(-maxs, maxs, Npts)
            posgrid = np.linspace(minp, maxp, Npts)

            mesh = np.meshgrid(posgrid, velgrid)
            meshgrid = np.concatenate([mesh[0].reshape(-1, 1), mesh[1].reshape(-1, 1)], axis=1)

            # value function
            w0_bar, phi_mesh = sess.run([QNet.w0_bar, QNet.phi], feed_dict={QNet.state: meshgrid})
            Qmesh = np.einsum('di,bdj->bj',  w0_bar, phi_mesh)
            Vmesh = np.max(Qmesh, axis=1)

            # plot last trajectory
            state_train = np.zeros([step, FLAGS.state_space])
            for k, experience in enumerate(tempbuffer.buffer):
                # [s, a, r, s', a*, d]
                state_train[k] = experience[0]

            # figure
            fig, ax = plt.subplots(ncols=1, figsize=[15, 5])
            im = ax.imshow(Vmesh.reshape(Npts, Npts), origin='lower', extent=[minp,maxp,-maxs,maxs])
            ax.set_aspect((maxp- minp)/(2* maxs))
            fig.colorbar(im)
            ax.set_xlim([minp, maxp])
            ax.set_ylim([-maxs, maxs])
            ax.set_xlabel('position')
            ax.set_ylabel('velocity')
            plt.scatter(state_train[:,0], state_train[:,1], s=5) # [pos, vel]
            plt.savefig(V0_dir + 'Epoch_' + str(episode) + '_step_' + str(step) + '_Reward')
            plt.close()

        if episode % 1000 == 0:
            log.info('Episode %3.d with time per episode %5.2f', episode, (time.time()- start))


    # reset buffers
    fullbuffer.reset()
    tempbuffer.reset()
