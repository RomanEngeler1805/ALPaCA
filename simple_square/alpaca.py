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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.16)

# General Hyperparameters
tf.flags.DEFINE_integer("batch_size", 2, "Batch size for training")
tf.flags.DEFINE_integer("action_space", 3, "Dimensionality of action space")
tf.flags.DEFINE_integer("state_space", 9, "Dimensionality of state space")
tf.flags.DEFINE_integer("hidden_space", 64, "Dimensionality of hidden space")
tf.flags.DEFINE_integer("latent_space", 8, "Dimensionality of latent space")
tf.flags.DEFINE_float("gamma", 0.9, "Discount factor")

tf.flags.DEFINE_float("learning_rate", 3e-3, "Initial learning rate")
tf.flags.DEFINE_float("lr_drop", 1.001, "Drop of learning rate per episode")

tf.flags.DEFINE_float("prior_precision", 0.2, "Prior precision (1/var)")
tf.flags.DEFINE_float("noise_precision", 10., "Noise precision (1/var)") # increase
tf.flags.DEFINE_float("noise_precmax", 30, "Maximum noise precision (1/var)")
tf.flags.DEFINE_integer("noise_Ndrop", 50, "Increase noise precision every N steps")
tf.flags.DEFINE_float("noise_precstep", 1.0001, "Step of noise precision s*=ds")

tf.flags.DEFINE_integer("split_N", 10000, "Increase split ratio every N steps")
tf.flags.DEFINE_float("split_ratio", 0.20, "Initial split ratio for conditioning")
tf.flags.DEFINE_integer("update_freq_post", 4, "Update frequency of posterior and sampling of new policy")

tf.flags.DEFINE_integer("kl_freq", 100, "Update kl divergence comparison")
tf.flags.DEFINE_float("kl_lambda", 10., "Weight for Kl divergence in loss")

tf.flags.DEFINE_integer("N_episodes", 6000, "Number of episodes")
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

                # State Value Fcn -----------------------------------------------------------
                if (step == 0 or step == FLAGS.L_episode-1 ) and n == 0 and episode % FLAGS.save_frequency == 0:
                    state_train = np.zeros([step + 1, FLAGS.state_space])

                    # fill array
                    for k, experience in enumerate(tempbuffer.buffer):
                        state_train[k] = experience[0]

                    tar = env.target  # target location as in array notation i.e. tar[0] downwards, tar[1] rightwards
                    # state value
                    V_TS = np.zeros([FLAGS.state_space, 3])
                    dV_TS = np.zeros([FLAGS.state_space, 3])
                    for i in range(FLAGS.state_space):
                        ss = np.zeros([FLAGS.state_space])  # loop over one-hot encoding
                        ss[i] = 1
                        w0_bar, L0, Sigma_e, phi = sess.run([QNet.w0_bar, QNet.L0, QNet.Sigma_e, QNet.phi],
                                                            feed_dict={QNet.state: ss.reshape(1, -1),
                                                                       QNet.nprec: noise_precision})

                        try:
                            Qout = np.dot(np.transpose(wt_bar), phi[0])
                            dQout = np.einsum('ia,ij,ja->a', phi[0], Lt_inv, phi[0])
                        except:
                            Qout = np.dot(np.transpose(w0_bar), phi[0])
                            dQout = np.einsum('ia,ij,ja->a', phi[0], np.linalg.inv(L0), phi[0])

                        #
                        #arg_Qmax = np.argmax(Qout, axis=1)

                        V_TS[i] = Qout#[0,arg_Qmax]
                        dV_TS[i] = dQout#[arg_Qmax]

                    # plotting
                    plot_Valuefcn(V_TS, dV_TS, tar, V_M_dir + 'Epoch_' + str(episode) + '_Step_' + str(step), state_train)
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

                    # State Value Fcn -----------------------------------------------------------
                    if (episode) % FLAGS.save_frequency == 0 and n == 0:
                        state_train = np.append(state_train, next_state.reshape(1, -1), axis=0)
                        tar = env.target  # target location as in array notation i.e. tar[0] downwards, tar[1] rightwards
                        # state value
                        V_TS = np.zeros([FLAGS.state_space,3])
                        dV_TS = np.zeros([FLAGS.state_space,3])
                        for i in range(FLAGS.state_space):
                            ss = np.zeros([FLAGS.state_space])  # loop over one-hot encoding
                            ss[i] = 1
                            # i/5 downwards, i%5 rightwards
                            # if i / 5 == tar[0] and i % 5 == tar[1]:  # add reward at target location
                            #    ss[FLAGS.state_space - 1] = 1.
                            Sigma_e, phi = sess.run([QNet.Sigma_e, QNet.phi],
                                                    feed_dict={QNet.state: ss.reshape(1, -1),
                                                               QNet.nprec: noise_precision})
                            Qout = np.dot(np.transpose(wt_bar), phi[0])
                            dQout = np.einsum('ia,ij,ja->a', phi[0], Lt_inv, phi[0])

                            arg_Qmax = np.argmax(Qout, axis=1)

                            V_TS[i] = Qout#[0,arg_Qmax]
                            dV_TS[i] = dQout#[arg_Qmax]

                        # plotting
                        plot_Valuefcn(V_TS, dV_TS, tar, V_M_dir + 'Epoch_' + str(episode) + '_Step_' + str(step+1), state_train)

                    # -----------------------------------------------------------------------

                # update state, and counters
                state = next_state.copy()
                global_index += 1
                step += 1

                # -----------------------------------------------------------------------
            del wt_bar, Lt_inv

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
            N_eval = 50

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

            reward_eval = tf.Summary(value=[tf.Summary.Value(tag='Reward Eval', simple_value=np.sum(reward_eval)/ N_eval)])
            summary_writer.add_summary(reward_eval, episode)
            summary_writer.flush()

    # write reward to file
    df = pd.DataFrame(reward_episode)
    df.to_csv(reward_dir+'reward_per_episode', index=False)

    # reset buffers
    fullbuffer.reset()
    tempbuffer.reset()
