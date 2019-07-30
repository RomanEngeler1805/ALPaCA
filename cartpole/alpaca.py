'''
Class structure for ALPaCA meta-RL code
'''

import numpy as np
import tensorflow as tf
import os
import time
import logging
import matplotlib.pyplot as plt
import pandas as pd
import sys
from matplotlib import ticker

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.16)

# General Hyperparameters
tf.flags.DEFINE_integer("batch_size", 2, "Batch size for training")
tf.flags.DEFINE_integer("action_space", 2, "Dimensionality of action space")
tf.flags.DEFINE_integer("state_space", 4, "Dimensionality of state space")
tf.flags.DEFINE_integer("hidden_space", 64, "Dimensionality of hidden space")
tf.flags.DEFINE_integer("latent_space", 8, "Dimensionality of latent space")
tf.flags.DEFINE_float("gamma", 0.95, "Discount factor")

tf.flags.DEFINE_float("learning_rate", 2e-3, "Initial learning rate")
tf.flags.DEFINE_float("lr_drop", 1.001, "Drop of learning rate per episode")

tf.flags.DEFINE_float("prior_precision", 0.1, "Prior precision (1/var)")
tf.flags.DEFINE_float("noise_precision", 0.1, "Noise precision (1/var)")
tf.flags.DEFINE_float("noise_precmax", 5, "Maximum noise precision (1/var)")
tf.flags.DEFINE_integer("noise_Ndrop", 1, "Increase noise precision every N steps")
tf.flags.DEFINE_float("noise_precstep", 1.0001, "Step of noise precision s*=ds")

tf.flags.DEFINE_integer("split_N", 10000, "Increase split ratio every N steps")
tf.flags.DEFINE_float("split_ratio", 0., "Initial split ratio for conditioning")
tf.flags.DEFINE_integer("update_freq_post", 3000, "Update frequency of posterior and sampling of new policy")

tf.flags.DEFINE_integer("kl_freq", 100, "Update kl divergence comparison")
tf.flags.DEFINE_float("kl_lambda", 10., "Weight for Kl divergence in loss")

tf.flags.DEFINE_integer("N_episodes", 6000, "Number of episodes")
tf.flags.DEFINE_integer("N_tasks", 2, "Number of tasks")
tf.flags.DEFINE_integer("L_episode", 600, "Length of episodes")

tf.flags.DEFINE_float("tau", 1., "Update speed of target network")
tf.flags.DEFINE_integer("update_freq_target", 100, "Update frequency of target network")

tf.flags.DEFINE_integer("replay_memory_size", 1000, "Size of replay memory")
tf.flags.DEFINE_integer("iter_amax", 1, "Number of iterations performed to determine amax")
tf.flags.DEFINE_integer("save_frequency", 200, "Store images every N-th episode")
<<<<<<< HEAD
tf.flags.DEFINE_float("regularizer", 0.1, "Regularization parameter")
=======
tf.flags.DEFINE_float("regularizer", 0.001, "Regularization parameter")
>>>>>>> 0a7d87278bf10ca98e3f2b73319ce43f4b76ccd3
tf.flags.DEFINE_string('non_linearity', 'relu', 'Non-linearity used in encoder')

tf.flags.DEFINE_integer("random_seed", 1234, "Random seed for numpy and tensorflow")
tf.flags.DEFINE_integer("sample_mass", 1, "If pole mass is sampled or fixed")
tf.flags.DEFINE_integer("sample_length", 0, "If pole length is sampled or fixed")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)

np.random.seed(FLAGS.random_seed)
tf.set_random_seed(FLAGS.random_seed)

from cartpole import CartPoleEnv
from QNetwork import QNetwork

sys.path.insert(0, './..')
from replay_buffer import replay_buffer

from prioritized_memory import Memory

def eGreedyAction(x, epsilon=0.):
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
split_ratio = FLAGS.split_ratio

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

dV_dir = 'figures/' + time.strftime('%H-%M-%d_%m-%y') + '/dV/'
if not os.path.exists(dV_dir):
    os.makedirs(dV_dir)

# initialize replay memory and model
fullbuffer = Memory(FLAGS.replay_memory_size) #replay_buffer(FLAGS.replay_memory_size) # large buffer to store all experience
tempbuffer = replay_buffer(FLAGS.L_episode) # buffer for episode
log.info('Build Tensorflow Graph')

# initialize environment
env = CartPoleEnv(FLAGS.sample_mass, FLAGS.sample_length) #gym.make('MountainCar-v0')

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    QNet = QNetwork(FLAGS, scope='QNetwork')  # neural network
    Qtarget = QNetwork(FLAGS, scope='TargetNetwork')

    # session
    init = tf.global_variables_initializer()
    sess.run(init)

    # DKL with old values (limit rate of change)
    w0_bar_old = np.zeros([2, FLAGS.latent_space, 1])
    L0_asym_old = np.zeros([2, FLAGS.latent_space])

    w0_bar_old[0], L0_asym_old[0] = sess.run([QNet.w0_bar, QNet.L0_asym], {QNet.episode: 0})
    w0_bar_old[1] = w0_bar_old[0]
    L0_asym_old[1] = L0_asym_old[0]

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

    gradBuffer = sess.run(QNet.tvars) # get shapes of tensors

    for idx in range(len(gradBuffer)):
        gradBuffer[idx] *= 0

    # loss buffers to visualize in tensorboard
    lossBuffer = 0.
    loss0Buffer = 0.
    loss1Buffer = 0.
    loss2Buffer = 0.
    loss3Buffer = 0.
    lossregBuffer = 0.

    # report mean reward per episode
    reward_episode = []

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

            # sample theta (i.e. bandit)
            env._sample_env()

            # resample state
            state = env._sample_state()

            # sample w from prior
            sess.run([QNet.sample_prior])

            # loop steps
            step = 0

            while step < FLAGS.L_episode:

                # take a step
                phi_online = sess.run(QNet.phi, feed_dict={QNet.state: state.reshape(-1,FLAGS.state_space),
                                                           QNet.episode: episode})
                wt_online = sess.run(QNet.wt)
                Qval = np.einsum('jm,bjk->bk', wt_online, phi_online)

                action = eGreedyAction(Qval, eps)
                next_state, reward, done, _ = env._step(action)

                # store experience in memory
                new_experience = [state, action, reward, next_state, done]
                tempbuffer.add(new_experience)

                # actual reward
                rw.append(reward)

                # update state, and counters
                state = next_state.copy()
                global_index += 1
                step += 1

                if done == 1:
                    break

                # update posterior
                # TODO: could speed up by iteratively adding
                if (step+1) % FLAGS.update_freq_post == 0 and (step+1) <= np.int(split_ratio* FLAGS.L_episode):
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
                    _, wt_bar = sess.run([QNet.sample_post, QNet.wt_bar],
                             feed_dict={QNet.context_state: state_train, QNet.context_action: action_train,
                                        QNet.context_reward: reward_train, QNet.context_state_next: next_state_train,
                                        QNet.context_done: done_train,
                                        QNet.nprec: noise_precision, QNet.episode: episode})

                # -----------------------------------------------------------------------
            # append episode buffer to large buffer
            fullbuffer.add(tempbuffer.buffer)

        # reward in episode
        reward_episode.append(np.sum(np.array(rw))/ FLAGS.N_tasks)

        if episode % 1000 == 0:
            log.info('Episode %3.d with R %3.d', episode, np.sum(rw))

        # learning rate schedule
        if learning_rate > 1e-6:
            learning_rate /= FLAGS.lr_drop

        if noise_precision < FLAGS.noise_precmax and episode % FLAGS.noise_Ndrop == 0:
            noise_precision *= FLAGS.noise_precstep

        if episode % FLAGS.split_N == 0 and episode > 0:
            split_ratio = np.min([split_ratio+ 0.01, 0.15])

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
            split = np.int(split_ratio* L_experience* np.random.rand())
            #split = np.int(np.min([L_experience-1, np.max([FLAGS.update_freq_post, split_ratio* L_experience])* np.random.rand()])) #np.int(split_ratio* L_experience* np.random.rand())

            train = np.arange(0, split)
            valid = np.arange(split, L_experience)

            # split in train and validation set
            #train = np.random.choice(np.arange(L_experience), np.int(split_ratio* L_experience), replace=False)  # mixed
            #valid = np.setdiff1d(np.arange(L_experience), train)
            #valid = np.random.choice(valid, np.min([len(valid), np.int(0.3*L_experience)]), replace=False)

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

            # select amax from online network
            amax_online = sess.run(QNet.max_action,
                                   feed_dict={QNet.context_state: state_train, QNet.context_action: action_train,
                                              QNet.context_reward: reward_train,
                                              QNet.context_state_next: next_state_train,
                                              QNet.state: state_valid, QNet.state_next: next_state_valid,
                                              QNet.nprec: noise_precision, QNet.episode: episode})


            # evaluate target model
            Qmax_target = sess.run(Qtarget.Qmax,
                feed_dict={Qtarget.context_state: state_train, Qtarget.context_action: action_train,
                           Qtarget.context_reward: reward_train, Qtarget.context_state_next: next_state_train,
                           Qtarget.state: state_valid, Qtarget.state_next: next_state_valid,
                           Qtarget.amax_online: amax_online, Qtarget.episode: episode,
                           Qtarget.nprec: noise_precision})

            # update model
            grads, loss0, loss1, loss2, loss3, loss_reg, loss, summaries_encodinglayer = sess.run(
                [QNet.gradients, QNet.loss0, QNet.loss1, QNet.loss2, QNet.loss3, QNet.loss_reg, QNet.loss, QNet.summaries_encodinglayer],
                feed_dict={QNet.context_state: state_train, QNet.context_action: action_train,
                           QNet.context_reward: reward_train, QNet.context_state_next: next_state_train,
                           QNet.state: state_valid, QNet.action: action_valid,
                           QNet.reward: reward_valid, QNet.state_next: next_state_valid,
                           QNet.done: done_valid, QNet.Qmax_target: Qmax_target,
                           QNet.amax_online: amax_online, QNet.episode: episode,
                           QNet.lr_placeholder: learning_rate, QNet.nprec: noise_precision,
                           QNet.episode: episode,
                           QNet.w0_bar_old: w0_bar_old[0], QNet.L0_asym_old: L0_asym_old[0]})

            for idx, grad in enumerate(grads): # grad[0] is gradient and grad[1] the variable itself
                gradBuffer[idx] += (grad[0]/ batch_size)

            lossBuffer += loss
            loss0Buffer += loss0
            loss1Buffer += loss1
            loss2Buffer += loss2
            loss3Buffer += loss3
            lossregBuffer += loss_reg

        # update summary
        feed_dict = dictionary = dict(zip(QNet.gradient_holders, gradBuffer))
        feed_dict.update({QNet.lr_placeholder: learning_rate})

        # reduce summary size
        if episode % 10 == 0:
            # update summary
            _, summaries_gradvar = sess.run([QNet.updateModel, QNet.summaries_gradvar], feed_dict=feed_dict)
            summaries_var = sess.run(Qtarget.summaries_var)

            loss_summary = tf.Summary(value=[tf.Summary.Value(tag='Loss', simple_value=(lossBuffer / batch_size))])
            loss0_summary = tf.Summary(value=[tf.Summary.Value(tag='TD_Loss', simple_value=(loss0Buffer / batch_size))])
            loss1_summary = tf.Summary(value=[tf.Summary.Value(tag='TDW_Loss', simple_value=(loss1Buffer/ batch_size))])
            loss2_summary = tf.Summary(value=[tf.Summary.Value(tag='Sig_Loss', simple_value=(loss2Buffer / batch_size))])
            loss3_summary = tf.Summary(value=[tf.Summary.Value(tag='KL_Loss', simple_value=(loss3Buffer / batch_size))])
            lossreg_summary = tf.Summary(value=[tf.Summary.Value(tag='Regularization_Loss', simple_value=(lossregBuffer / batch_size))])
            reward_summary = tf.Summary(value=[tf.Summary.Value(tag='Episodic Reward', simple_value=np.sum(np.array(rw))/(FLAGS.N_tasks))])
            learning_rate_summary = tf.Summary(value=[tf.Summary.Value(tag='Learning rate', simple_value=learning_rate)])
            noise_summary = tf.Summary(value=[tf.Summary.Value(tag='Noise variance', simple_value=1./noise_precision)])

            summary_writer.add_summary(loss_summary, episode)
            summary_writer.add_summary(loss0_summary, episode)
            summary_writer.add_summary(loss1_summary, episode)
            summary_writer.add_summary(loss2_summary, episode)
            summary_writer.add_summary(loss3_summary, episode)
            summary_writer.add_summary(lossreg_summary, episode)
            summary_writer.add_summary(reward_summary, episode)
            summary_writer.add_summary(summaries_var, episode)
            summary_writer.add_summary(summaries_gradvar, episode)
            summary_writer.add_summary(summaries_encodinglayer, episode)
            summary_writer.add_summary(learning_rate_summary, episode)
            summary_writer.add_summary(noise_summary, episode)

            summary_writer.flush()
        else:
            _ = sess.run([QNet.updateModel], feed_dict=feed_dict)


        # reset buffers
        for idx in range(len(gradBuffer)):
            gradBuffer[idx] *= 0

        lossBuffer *= 0.
        loss0Buffer *= 0.
        loss1Buffer *= 0.
        loss2Buffer *= 0.
        loss3Buffer *= 0.
        lossregBuffer *= 0.

        # increase the batch size after the first episode. Would allow N_tasks < batch_size due to buffer
        if episode < 2:
            batch_size *= 2

        # kl divergence updates
        if episode % FLAGS.kl_freq == 0:
            w0_bar_old[0] = w0_bar_old[1]
            L0_asym_old[0] = L0_asym_old[1]
            w0_bar_old[1], L0_asym_old[1] = sess.run([QNet.w0_bar, QNet.L0_asym], {QNet.episode: 0})

        # ===============================================================
        # update target network
        if episode % FLAGS.update_freq_target == 0:
            vars_modelQ = sess.run(QNet.tvars)
            feed_dict = dictionary = dict(zip(Qtarget.variable_holders, vars_modelQ))
            feed_dict.update({Qtarget.tau: FLAGS.tau})
            sess.run(Qtarget.copyParams, feed_dict=feed_dict)

        # ===============================================================
        # save model
        #if episode > 0 and episode % 1000 == 0:
          # Save a checkpoint
          #log.info('Save model snapshot')

          #filename = os.path.join(saver_dir, 'model')
          #saver.save(sess, filename, global_step=episode, write_meta_graph=False)

        # ================================================================
        # print to console
        if episode % FLAGS.save_frequency == 0:
            log.info('Episode %3.d with R %3.d', episode, np.sum(rw))

            print('Reward in Episode ' + str(episode)+  ':   '+ str(np.sum(rw)))
            print('Learning_rate: ' + str(np.round(learning_rate, 5)) + ', Nprec: ' + str(np.round(noise_precision, 4)) + ', Split ratio: ' + str(np.round(split_ratio, 2)))

            # plot trajectory ----------------------------------------------
            state_train = np.zeros([step, FLAGS.state_space])
            for k, experience in enumerate(tempbuffer.buffer):
                # [s, a, r, s', a*, d]
                state_train[k] = experience[0]

            # limits
            xlim = env.x_threshold
            alim = env.theta_threshold_radians

            plt.figure()
            plt.plot(state_train[:,0], state_train[:,2], marker='o', markersize=4, color='b')
            plt.plot(state_train[0, 0], state_train[0, 2], 'ro')
            plt.xlabel('pos [m]')
            plt.ylabel('angle [rad]')
            plt.plot([-xlim, xlim], [alim, alim], '--r')
            plt.plot([-xlim, xlim], [-alim, -alim], '--r')
            plt.plot([xlim, xlim], [-alim, alim], '--r')
            plt.plot([-xlim, -xlim], [-alim, alim], '--r')

            plt.savefig(Vt_dir + 'Epoch_' + str(episode) + '_step_' + str(step)+ '_mass_'+ str(np.int(1000*env.masspole)) + '_Reward')
            plt.close()

            # plot value function --------------------------------------------
            # fix x-position and x-acceleration study behaviour as function of pole state
            fig, ax = plt.subplots(ncols=3, nrows=3, figsize=[20, 12])

            Npts = 50
            max_theta = 0.42
            max_thetadot = 1.

            poletheta = np.linspace(-max_theta, +max_theta, Npts)
            polethetadot = np.linspace(-1., +1., Npts)

            polemesh = np.meshgrid(poletheta, polethetadot)
            for row, xpos in enumerate([2.4, 0, 2.4]):
                for col, xacc in enumerate([-10., 0., 10.]):
                    # evaluate value function

                    meshgrid = np.concatenate([xpos* np.ones([Npts*Npts, 1]),
                                               xacc* np.ones([Npts*Npts, 1]),
                                               polemesh[0].reshape(-1, 1),
                                               polemesh[1].reshape(-1, 1)], axis=1)

                    # value function
                    w0_bar, phi_mesh = sess.run([QNet.w0_bar, QNet.phi], feed_dict={QNet.state: meshgrid, QNet.episode: 0})
                    Qmesh = np.einsum('di,bda->ba', w0_bar, phi_mesh)
                    Vmesh = np.max(Qmesh, axis=1)

                    im = ax[row, col].imshow(Vmesh.reshape(Npts, Npts), origin='lower', extent=[-max_theta, max_theta, -max_thetadot, max_thetadot])
                    ax[row, col].set_aspect(max_theta/ max_thetadot)
                    #cb = fig.colorbar(im, ax=ax[row, col], shrink=0.74, orientation="horizontal", pad=0.2)
                    #tick_locator = ticker.MaxNLocator(nbins=4)
                    #cb.locator = tick_locator
                    #cb.update_ticks()
                    ax[row, col].set_xlim([-max_theta, max_theta])
                    ax[row, col].set_ylim([-max_thetadot, max_thetadot])
                    ax[row, col].set_xlabel('theta')
                    ax[row, col].set_ylabel('theta dot')
                    #ax[0].title.set_text('Mean')

            plt.savefig(V0_dir + 'Epoch_' + str(episode))
            plt.close()

            # plot uncertainty in Value function -------------------------------
            meshgrid = np.concatenate([0. * np.ones([Npts * Npts, 1]),
                                       0. * np.ones([Npts * Npts, 1]),
                                       polemesh[0].reshape(-1, 1),
                                       polemesh[1].reshape(-1, 1)], axis=1)

            # value function
            w0_bar, L0, phi_mesh = sess.run([QNet.w0_bar, QNet.L0, QNet.phi], feed_dict={QNet.state: meshgrid, QNet.episode: 0})
            Qmesh = np.einsum('di,bda->ba',  w0_bar, phi_mesh)
            dQmesh = np.einsum('bia,ij,bja->ba', phi_mesh, np.linalg.inv(L0), phi_mesh)

            Vmesh = np.max(Qmesh, axis=1)

            Vmaxplot = np.max(Qmesh+ dQmesh, axis=1)
            Vminplot = np.min(Qmesh- dQmesh, axis=1)
            dV = Vmaxplot- Vminplot

            # figure
            fig, ax = plt.subplots(ncols=4, figsize=[16, 5])

            im = ax[0].imshow(Vmesh.reshape(Npts, Npts), origin='lower', extent=[-max_theta, max_theta,-max_thetadot, max_thetadot])
            ax[0].set_aspect(max_theta/max_thetadot)
            cb = fig.colorbar(im, ax=ax[0], shrink=0.74, orientation="horizontal", pad=0.2)
            tick_locator = ticker.MaxNLocator(nbins=4)
            cb.locator = tick_locator
            cb.update_ticks()
            ax[0].set_xlim([-max_theta, max_theta])
            ax[0].set_ylim([-max_thetadot, max_thetadot])
            ax[0].set_xlabel('theta')
            ax[0].set_ylabel('theta dot')
            ax[0].title.set_text('Mean')

            im = ax[1].imshow(Vmesh.reshape(Npts, Npts)+ dV.reshape(Npts, Npts), origin='lower', extent=[-max_theta, max_theta,-max_thetadot, max_thetadot])
            ax[1].set_aspect(max_theta/max_thetadot)
            cb = fig.colorbar(im, ax=ax[1], shrink=0.74, orientation="horizontal", pad=0.2)
            tick_locator = ticker.MaxNLocator(nbins=4)
            cb.locator = tick_locator
            cb.update_ticks()
            ax[1].set_xlim([-max_theta, max_theta])
            ax[1].set_ylim([-max_thetadot, max_thetadot])
            ax[1].set_xlabel('theta')
            ax[1].set_ylabel('theta dot')
            ax[1].title.set_text('Mean+ Stdv')

            im = ax[2].imshow(Vmesh.reshape(Npts, Npts)- dV.reshape(Npts, Npts), origin='lower', extent=[-max_theta, max_theta,-max_thetadot, max_thetadot])
            ax[2].set_aspect(max_theta/max_thetadot)
            cb = fig.colorbar(im, ax=ax[2], shrink=0.74, orientation="horizontal", pad=0.2)
            tick_locator = ticker.MaxNLocator(nbins=4)
            cb.locator = tick_locator
            cb.update_ticks()
            ax[2].set_xlim([-max_theta, max_theta])
            ax[2].set_ylim([-max_thetadot, max_thetadot])
            ax[2].set_xlabel('theta')
            ax[2].set_ylabel('theta dot')
            ax[2].title.set_text('Mean- Stdv')

            im = ax[3].imshow(dV.reshape(Npts, Npts), origin='lower', extent=[-max_theta, max_theta,-max_thetadot, max_thetadot])
            ax[3].set_aspect(max_theta/max_thetadot)
            cb = fig.colorbar(im, ax=ax[3], shrink=0.74, orientation="horizontal", pad=0.2)
            tick_locator = ticker.MaxNLocator(nbins=4)
            cb.locator = tick_locator
            cb.update_ticks()
            ax[3].set_xlim([-max_theta, max_theta])
            ax[3].set_ylim([-max_thetadot, max_thetadot])
            ax[3].set_xlabel('theta')
            ax[3].set_ylabel('theta dot')
            ax[3].title.set_text('Stdv')

            plt.savefig(dV_dir + 'Epoch_' + str(episode) + '_step_' + str(step) + '_Reward')
            plt.close()

            # evaluate on 10 different initial positions with w0_bar
            # loop tasks --------------------------------------------------------------------
            reward_valid = np.zeros([10])
            for n in range(10):
                # sample environment
                env._sample_env()

                # resample state
                state = env._sample_state()

                # loop steps
                step = 0

                while step < FLAGS.L_episode:
                    # take a step
                    Qval = sess.run([QNet.Qmean], feed_dict={QNet.state: state.reshape(-1, FLAGS.state_space), QNet.episode: 0}) # mean policy
                    action = eGreedyAction(Qval, eps)
                    next_state, reward, done, _ = env._step(action)

                    # actual reward
                    reward_valid[n]+= reward

                    # update state, and counters
                    state = next_state.copy()
                    step += 1

                    if done == 1:
                        break

            valid_reward_summary = tf.Summary(value=[tf.Summary.Value(tag='Validation Reward', simple_value=np.mean(reward_valid))])
            summary_writer.add_summary(valid_reward_summary, episode)
            summary_writer.flush()

        if episode % 1000 == 0:
            log.info('Episode %3.d with time per episode %5.2f', episode, (time.time()- start))

    # write reward to file
    df = pd.DataFrame(reward_episode)
    df.to_csv(reward_dir+'reward_per_episode', index=False)

    # reset buffers
    fullbuffer.reset()
    tempbuffer.reset()
