'''
Class structure for ALPaCA meta-RL code
'''

import numpy as np
import tensorflow as tf
import os
import time
import logging
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pandas as pd
from matplotlib import ticker
import sys
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.16)

# General Hyperparameters
tf.flags.DEFINE_integer("batch_size", 2, "Batch size for training")
tf.flags.DEFINE_integer("action_space", 3, "Dimensionality of action space")
tf.flags.DEFINE_integer("state_space", 9, "Dimensionality of state space")
tf.flags.DEFINE_integer("hidden_space", 64, "Dimensionality of hidden space")
tf.flags.DEFINE_integer("latent_space", 8, "Dimensionality of latent space")
tf.flags.DEFINE_float("gamma", 0.95, "Discount factor")

tf.flags.DEFINE_float("learning_rate", 3e-4, "Initial learning rate")
tf.flags.DEFINE_float("lr_drop", 1.0001, "Drop of learning rate per episode")

tf.flags.DEFINE_float("prior_precision", 0.2, "Prior precision (1/var)")
tf.flags.DEFINE_float("noise_precision", 10., "Noise precision (1/var)") # increase
tf.flags.DEFINE_float("noise_precmax", 30, "Maximum noise precision (1/var)")
tf.flags.DEFINE_integer("noise_Ndrop", 50, "Increase noise precision every N steps")
tf.flags.DEFINE_float("noise_precstep", 1.0001, "Step of noise precision s*=ds")

tf.flags.DEFINE_integer("split_N", 10000, "Increase split ratio every N steps")
tf.flags.DEFINE_float("split_ratio", 0.2, "Initial split ratio for conditioning")
tf.flags.DEFINE_integer("update_freq_post", 4, "Update frequency of posterior and sampling of new policy")

tf.flags.DEFINE_integer("kl_freq", 100, "Update kl divergence comparison")
tf.flags.DEFINE_float("kl_lambda", 10., "Weight for Kl divergence in loss")

tf.flags.DEFINE_integer("N_episodes", 6001, "Number of episodes")
tf.flags.DEFINE_integer("N_tasks", 2, "Number of tasks")
tf.flags.DEFINE_integer("L_episode", 50, "Length of episodes")

tf.flags.DEFINE_float("tau", 1., "Update speed of target network")
tf.flags.DEFINE_integer("update_freq_target", 1, "Update frequency of target network")

tf.flags.DEFINE_integer("replay_memory_size", 1000, "Size of replay memory")
tf.flags.DEFINE_integer("iter_amax", 1, "Number of iterations performed to determine amax")
tf.flags.DEFINE_integer("save_frequency", 200, "Store images every N-th episode")
tf.flags.DEFINE_float("regularizer", 0.01, "Regularization parameter")
tf.flags.DEFINE_string('non_linearity', 'sigm', 'Non-linearity used in encoder')

tf.flags.DEFINE_integer("random_seed", 1234, "Random seed for numpy and tensorflow")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)

np.random.seed(FLAGS.random_seed)
tf.set_random_seed(FLAGS.random_seed)

from environment import environment
from QNetwork import QNetwork

sys.path.insert(0, './..')
from replay_buffer import replay_buffer

def eGreedyAction(x, epsilon=0.):
    ''' select next action according to epsilon-greedy algorithm '''
    if np.random.rand() >= epsilon:
        action = np.argmax(x)
    else:
        action = np.random.randint(FLAGS.action_space)

    return action


def plot_Valuefcn(Valuefcn, dValuefcn, target, save_path, states=np.array([])):
    #
    fig, ax = plt.subplots(figsize=[8, 10], nrows=3)
    im0 = ax[0].imshow(np.transpose(Valuefcn), aspect='auto')
    im1 = ax[1].imshow(np.transpose(dValuefcn), aspect='auto')
    im2 = ax[2].imshow(np.max(Valuefcn, axis=1).reshape(1,-1))
    #im2 = ax[1].imshow(dValuefcn.reshape(1, FLAGS.state_space))

    ax[2].plot(target, 0, 'ro', markersize=20)

    if len(state)> 0:
        pos = np.argmax(states, axis=1)
        ax[2].plot(pos, np.zeros(len(pos)), 'bo', markersize=20)  # imshow makes y-axis pointing downwards

    cb = fig.colorbar(im0, ax=ax[0], shrink=1.0, orientation="horizontal", pad=0.2)
    cb = fig.colorbar(im1, ax=ax[1], shrink=1.0, orientation="horizontal", pad=0.2)
    cb = fig.colorbar(im2, ax=ax[2], shrink=1.0, orientation="horizontal", pad=0.2)

    ax[0].title.set_text('Mean Q Function')
    ax[1].title.set_text('Stdv Q Function')
    ax[2].title.set_text('Mean V Function')

    plt.savefig(save_path)
    plt.close()

def plot_Valuefcn_confidence(Valuefcn, dValuefcn, target, save_path, states=np.array([]), steps=7):
    # normalize
    # Valuefcn/= np.mean(Valuefcn, axis=1)[:, None]
    # dValuefcn/= np.mean(Valuefcn, axis=1)[:, None]

    #
    fig, ax = plt.subplots(figsize=[8, 4], nrows=5)
    for s in range(4):
        ax[s].plot(np.arange(9), Valuefcn[s], c='b')
        lower = Valuefcn[s]- 1.96* np.sqrt(dValuefcn[s])
        upper = Valuefcn[s]+ 1.96* np.sqrt(dValuefcn[s])
        ax[s].fill_between(np.arange(9), lower, upper, alpha=0.5)
        ax[s].plot([target, target], [np.min(lower), np.max(upper)], c='r')
        pos = np.argmax(states[s])
        ax[s].plot([pos, pos], [np.min(lower), np.max(upper)], c='b')  # imshow makes y-axis pointing downwards

        ax[s].set_xticks([])
        #ax[s].set_ylim([0.7* np.min(Valuefcn[s]), 1.3* np.max(Valuefcn[s])])
        #ax[s].set_yticks([])

    ax[4].plot(np.arange(9), Valuefcn[7], c='b')
    lower = Valuefcn[7] - 1.96 * np.sqrt(dValuefcn[7])
    upper = Valuefcn[7] + 1.96 * np.sqrt(dValuefcn[7])
    ax[4].fill_between(np.arange(9), lower, upper, alpha=0.5)
    ax[4].plot([target, target], [np.min(lower), np.max(upper)], c='r')
    pos = np.argmax(states[7])
    ax[4].plot([pos, pos], [np.min(lower), np.max(upper)], c='b')  # imshow makes y-axis pointing downwards

    ax[4].set_xticks([])
    #ax[4].set_ylim([0.7 * np.min(Valuefcn[7]), 1.3 * np.max(Valuefcn[7])])
    # ax[s].set_yticks([])

    plt.savefig(save_path)
    plt.close()

def plot_Value_Paper(Valuefcn, target, save_path, states=np.array([]), steps=7):
    #
    fig, ax = plt.subplots(figsize=[8, 6], nrows=5)
    for s in range(4):
        c_low = np.floor(np.min(Valuefcn[s]))
        c_up = np.ceil(np.max(Valuefcn[s]))
        im = ax[s].imshow(Valuefcn[s].reshape(1,-1))
        ax[s].plot(target, 0, 'ro', markersize=20)
        pos = np.argmax(states[s])
        ax[s].plot(pos, 0., 'bo', markersize=20)  # imshow makes y-axis pointing downwards
        ax[s].set_xticks([])
        ax[s].set_yticks([])
        im.set_clim(np.min(Valuefcn[s]), np.max(Valuefcn[s]))
        cb = fig.colorbar(im, ax=ax[s], shrink=0.55, orientation="vertical", pad=0.1, aspect=1, format='%.0f')
        cb.set_ticks([c_low+1, c_up-1])

    c_low = np.floor(1.1 * np.min(Valuefcn[7]))
    c_up = np.ceil(0.9 * np.max(Valuefcn[7]))
    im = ax[4].imshow(Valuefcn[7].reshape(1, -1), vmin=c_low, vmax=c_up)
    ax[4].plot(target, 0, 'ro', markersize=20)
    pos = np.argmax(states[7])
    ax[4].plot(pos, 0., 'bo', markersize=20)  # imshow makes y-axis pointing downwards
    ax[4].set_xticks([])
    ax[4].set_yticks([])
    im.set_clim(np.min(Valuefcn[7]), np.max(Valuefcn[7]))

    fig.subplots_adjust(right=0.8)
    cb = fig.colorbar(im, ax=ax[4], shrink=0.55, orientation="vertical", pad=0.1, aspect=1, format='%.0f')
    cb.set_ticks([c_low+1, c_up-1])

    plt.tight_layout()

    plt.savefig(save_path+'_'+str(np.int(np.min(Valuefcn)))+ '_'+ str(np.int(np.max(Valuefcn))))
    plt.close()




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
V_M_dir = './figures/'+ time.strftime('%H-%M-%d_%m-%y')+ '/Q_Fcn/'
if not os.path.exists(V_M_dir):
    os.makedirs(V_M_dir)

V_TS_dir = './figures/'+ time.strftime('%H-%M-%d_%m-%y')+ '/Q_TS_Fcn/'
if not os.path.exists(V_TS_dir):
    os.makedirs(V_TS_dir)

basis_fcn_dir = './figures/'+ time.strftime('%H-%M-%d_%m-%y')+ '/basis_fcn/'
if not os.path.exists(basis_fcn_dir):
    os.makedirs(basis_fcn_dir)

reward_dir = './figures/'+ time.strftime('%H-%M-%d_%m-%y')+ '/'
if not os.path.exists(reward_dir):
    os.makedirs(reward_dir)

data_dir = './data/'+ time.strftime('%H-%M-%d_%m-%y')+ '/'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# initialize replay memory and model
fullbuffer = replay_buffer(FLAGS.replay_memory_size) # large buffer to store all experience
tempbuffer = replay_buffer(FLAGS.L_episode) # buffer for episode
evalbuffer = replay_buffer(FLAGS.L_episode) # buffer for episode
log.info('Build Tensorflow Graph')

# initialize environment
env = environment(FLAGS.state_space)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    QNet = QNetwork(FLAGS, scope='QNetwork')  # neural network
    Qtarget = QNetwork(FLAGS, scope='TargetNetwork')

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

    gradBuffer = sess.run(QNet.tvars) # get shapes of tensors

    for idx in range(len(gradBuffer)):
        gradBuffer[idx] *= 0

    # loss buffers to visualize in tensorboard
    lossBuffer = 0.

    # report mean reward per episode
    reward_episode = []

    # -----------------------------------------------------------------------------------
    # loop episodes
    print("Episodes...")
    log.info('Loop over episodes')
    for episode in range(FLAGS.N_episodes):

        # count reward
        rw = []

        # loop tasks --------------------------------------------------------------------
        for n in range(FLAGS.N_tasks):
            # initialize buffer
            tempbuffer.reset()

            # sample theta (i.e. bandit)
            env._sample_env()

            # resample state
            state = env._sample_state().copy()

            # sample w from prior
            sess.run([QNet.sample_prior])

            # loop steps
            step = 0

            Value_plot = np.zeros([FLAGS.L_episode, FLAGS.state_space])
            dValue_plot = np.zeros([FLAGS.L_episode, FLAGS.state_space])

            while step < FLAGS.L_episode:
                # take a step
                Qval = sess.run([QNet.Qout], feed_dict={QNet.state: state.reshape(-1,FLAGS.state_space)})
                action = eGreedyAction(Qval, eps)

                next_state, reward, done = env._step(action)

                # store experience in memory
                new_experience = [state, action, reward, next_state, done]
                tempbuffer.add(new_experience)

                # actual reward
                rw.append(reward)

                # State Value Fcn ----------------------------------------------------------- XXXXXXXX RE
                if n == 0 and episode % FLAGS.save_frequency == 0:
                    if step == 0:
                        state_plot = np.eye(9)
                        wt_bar, L0, phi_plot = sess.run([QNet.w0_bar, QNet.L0, QNet.phi],
                                                        feed_dict={QNet.state: state_plot, QNet.nprec: noise_precision})
                        Lt_inv = np.linalg.inv(L0)
                    # value function
                    Qout = np.einsum('i,bia->ba', wt_bar.reshape(-1), phi_plot)
                    dQout = np.einsum('bia,ij,bja->ba',phi_plot, Lt_inv, phi_plot)
                    Qout_max = np.argmax(Qout, axis=1)
                    Value_plot[step, :] = np.max(Qout, axis=1)
                    dValue_plot[step, :] = dQout[np.arange(len(Qout_max)), Qout_max]

                # -----------------------------------------------------------------------

                # update posterior
                if (step + 1) % FLAGS.update_freq_post == 0 and (step + 1) <= np.int(split_ratio * FLAGS.L_episode):
                    reward_train = np.zeros([step + 1, ])
                    state_train = np.zeros([step + 1, FLAGS.state_space])
                    next_state_train = np.zeros([step + 1, FLAGS.state_space])
                    action_train = np.zeros([step + 1, ])
                    done_train = np.zeros([step + 1, 1])

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
                                   QNet.nprec: noise_precision, QNet.is_online: True})

                    # -----------------------------------------------------------------------

                # update state, and counters
                state = next_state.copy()
                global_index += 1
                step += 1

                # -----------------------------------------------------------------------

            # plot value fcn
            if n == 0 and episode % FLAGS.save_frequency == 0:
                state_train = np.zeros([step + 1, FLAGS.state_space])
                # fill array
                for k, experience in enumerate(tempbuffer.buffer):
                    state_train[k] = experience[0]
                tar = env.target  # target location as in array notation i.e. tar[0] downwards, tar[1] rightwards
                # plotting
                plot_Value_Paper(Value_plot, tar, V_M_dir + 'Epoch_' + str(episode) + '_Step_' + str(step), state_train)

                data_name = data_dir+ 'episode_'+ str(episode)
                np.savez(data_name, Value_plot, state_train, tar)


                plot_Valuefcn_confidence(Value_plot, dValue_plot, tar, V_M_dir + 'Epoch_' + str(episode) + '_Step_' + str(step), state_train)

            # append episode buffer to large buffer
            fullbuffer.add(tempbuffer.buffer)

        # append reward
        reward_episode.append(np.sum(np.array(rw)))

        # Gradient descent
        for e in range(batch_size):

            # sample from larger buffer [s, a, r, s', d] with current experience not yet included
            experience = fullbuffer.sample(1)

            state_sample = np.zeros((FLAGS.L_episode, FLAGS.state_space))
            action_sample = np.zeros((FLAGS.L_episode,))
            reward_sample = np.zeros((FLAGS.L_episode,))
            next_state_sample = np.zeros((FLAGS.L_episode, FLAGS.state_space))
            done_sample = np.zeros((FLAGS.L_episode,))

            # fill arrays
            for k, (s0, a, r, s1, d) in enumerate(experience[0]):
                state_sample[k] = s0
                action_sample[k] = a
                reward_sample[k] = r
                next_state_sample[k] = s1
                done_sample[k] = d

                # split into context and prediction set
                split = np.int(split_ratio * FLAGS.L_episode * np.random.rand())

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

            # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            # select amax from online network
            amax_online = sess.run(QNet.max_action,
                                   feed_dict={QNet.context_state: state_train, QNet.context_action: action_train,
                                              QNet.context_reward: reward_train,
                                              QNet.context_state_next: next_state_train,
                                              QNet.state: state_valid, QNet.state_next: next_state_valid,
                                              QNet.lr_placeholder: learning_rate,
                                              QNet.nprec: noise_precision})

            # evaluate target model
            phi_max_target = sess.run(Qtarget.phi_max,
                                   feed_dict={Qtarget.context_state: state_train, Qtarget.context_action: action_train,
                                              Qtarget.context_reward: reward_train,
                                              Qtarget.context_state_next: next_state_train,
                                              Qtarget.state: state_valid, Qtarget.state_next: next_state_valid,
                                              Qtarget.lr_placeholder: learning_rate,
                                              Qtarget.nprec: noise_precision, Qtarget.amax_online: amax_online})

            # update model
            grads, loss0, Qdiff = sess.run(
                [QNet.gradients, QNet.loss, QNet.Qdiff],
                feed_dict={QNet.context_state: state_train, QNet.context_action: action_train,
                           QNet.context_reward: reward_train, QNet.context_state_next: next_state_train,
                           QNet.state: state_valid, QNet.action: action_valid,
                           QNet.reward: reward_valid, QNet.state_next: next_state_valid,
                           QNet.done: done_valid,
                           QNet.lr_placeholder: learning_rate, QNet.nprec: noise_precision,
                           QNet.phi_max_target: phi_max_target, QNet.amax_online: amax_online})

            # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


            for idx, grad in enumerate(grads): # grad[0] is gradient and grad[1] the variable itself
                gradBuffer[idx] += (grad[0]/ batch_size)

            lossBuffer += loss0


        # update summary
        feed_dict = dictionary = dict(zip(QNet.gradient_holders, gradBuffer))
        feed_dict.update({QNet.lr_placeholder: learning_rate})

        if episode > 1:
            # reduce summary size
            if episode % 10 == 0:
                # update summary
                _, summaries_gradvar = sess.run([QNet.updateModel, QNet.summaries_gradvar], feed_dict=feed_dict)

                loss_summary = tf.Summary(value=[tf.Summary.Value(tag='Loss', simple_value=(lossBuffer / batch_size))])
                reward_summary = tf.Summary(value=[tf.Summary.Value(tag='Episodic Reward', simple_value=np.sum(np.array(rw)))])

                learning_rate_summary = tf.Summary(value=[tf.Summary.Value(tag='Learning rate', simple_value=learning_rate)])

                summary_writer.add_summary(loss_summary, episode)
                summary_writer.add_summary(reward_summary, episode)
                summary_writer.add_summary(learning_rate_summary, episode)
                summary_writer.add_summary(summaries_gradvar, episode)

                summary_writer.flush()
            else:
                _ = sess.run([QNet.updateModel], feed_dict=feed_dict)


        # reset buffers
        for idx in range(len(gradBuffer)):
            gradBuffer[idx] *= 0

        lossBuffer *= 0.

        # increase the batch size after the first episode. Would allow N_tasks < batch_size due to buffer
        if episode < 2:
            batch_size *= 2

        # learning rate schedule
        if learning_rate > 5e-5:
            learning_rate /= FLAGS.lr_drop

        if noise_precision < FLAGS.noise_precmax and episode % FLAGS.noise_Ndrop == 0:
            noise_precision *= FLAGS.noise_precstep

        # ===============================================================
        # update target network
        if episode % FLAGS.update_freq_target == 0:
            vars_modelQ = sess.run(QNet.tvars)
            feed_dict = dictionary = dict(zip(Qtarget.variable_holders, vars_modelQ))
            feed_dict.update({Qtarget.tau: FLAGS.tau})
            sess.run(Qtarget.copyParams, feed_dict=feed_dict)

        # print to console
            # print to console
        if episode % FLAGS.save_frequency == 0:
            log.info('Episode %3.d with R %3.d', episode, np.sum(rw))

            print('Reward in Episode ' + str(episode) + ':   ' + str(np.sum(rw)))
            print('Learning_rate: ' + str(np.round(learning_rate, 5)) + ', Nprec: ' + str(noise_precision))

            # ===============================================================
            # evaluation
            if episode == 8000:
                N_eval = 10000
            else:
                N_eval = 100

            reward_eval = np.zeros([N_eval])

            for ne in range(N_eval):
                # initialize buffer
                evalbuffer.reset()
                # sample theta (i.e. bandit)
                env._sample_env()
                # resample state
                state = env._sample_state().copy()
                # sample w from prior
                sess.run([QNet.sample_prior])
                # loop steps
                step = 0

                while step < FLAGS.L_episode:
                    # take a step
                    Qval = sess.run([QNet.Qout], feed_dict={QNet.state: state.reshape(-1,FLAGS.state_space)})
                    action = eGreedyAction(Qval, eps)
                    next_state, reward, done = env._step(action)

                    # store experience in memory
                    new_experience = [state, action, reward, next_state, done]
                    evalbuffer.add(new_experience)
                    reward_eval[ne] += reward

                    if (step+1) % FLAGS.update_freq_post == 0 and (step+1) <= np.int(split_ratio* FLAGS.L_episode):
                        reward_train = np.zeros([step+1, ])
                        state_train = np.zeros([step+1, FLAGS.state_space])
                        next_state_train = np.zeros([step+1, FLAGS.state_space])
                        action_train = np.zeros([step+1, ])
                        done_train = np.zeros([step+1, 1])

                        # fill arrays
                        for k, experience in enumerate(evalbuffer.buffer):
                            # [s, a, r, s', a*, d]
                            state_train[k] = experience[0]
                            action_train[k] = experience[1]
                            reward_train[k] = experience[2]
                            next_state_train[k] = experience[3]
                            done_train[k] = experience[4]

                        # update
                        _ = sess.run([QNet.sample_post],
                                 feed_dict={QNet.context_state: state_train, QNet.context_action: action_train,
                                            QNet.context_reward: reward_train, QNet.context_state_next: next_state_train,
                                            QNet.context_done: done_train,
                                            QNet.nprec: noise_precision})

                    # update state, and counters
                    state = next_state.copy()
                    global_index += 1
                    step += 1

            rew_env = np.linspace(0, 10, 9)
            rew_opt = 1./2.* (np.sum(rew_env[3:])+ np.sum(rew_env[5:])+ 10* 10.)

            regret_eval_summary = tf.Summary(value=[tf.Summary.Value(tag='Regret Eval', simple_value=np.sum(reward_eval)/ N_eval/ rew_opt)])
            reward_eval_summary = tf.Summary(value=[tf.Summary.Value(tag='Reward Eval', simple_value=np.sum(reward_eval)/ N_eval)])
            summary_writer.add_summary(regret_eval_summary, episode)
            summary_writer.add_summary(reward_eval_summary, episode)
            summary_writer.flush()

            if episode == 8000:
                # write regret to file
                regret = np.sum(reward_eval.reshape(100, 100), axis=1)/ 100/ rew_opt
                regr = np.mean(regret)
                dregr = np.std(regret)/ 100
                file = open(saver_dir+ 'regret', 'a')
                file.write('Normalized Reward: \n')
                file.write('{:1.2f} +- {:1.4f}\n'.format(regr, dregr))
                file.close()

    # write reward to file
    df = pd.DataFrame(reward_episode)
    df.to_csv(reward_dir+'reward_per_episode', index=False)

    # reset buffers
    fullbuffer.reset()
    tempbuffer.reset()
