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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from bandit_environment import bandit_environment

import sys
sys.path.insert(0, './..')
from replay_buffer import replay_buffer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.16)

# General Hyperparameters
# general
tf.flags.DEFINE_integer("batch_size", 2, "Batch size for training")
tf.flags.DEFINE_float("gamma", 0., "Discount factor")
tf.flags.DEFINE_integer("N_episodes", 20000, "Number of episodes")
tf.flags.DEFINE_integer("N_tasks", 2, "Number of tasks")
tf.flags.DEFINE_integer("L_episode", 40, "Length of episodes")

# architecture
tf.flags.DEFINE_integer("hidden_space", 128, "Dimensionality of hidden space")
tf.flags.DEFINE_integer("latent_space", 8, "Dimensionality of latent space")
tf.flags.DEFINE_string('non_linearity', 'leaky_relu', 'Non-linearity used in encoder')

# domain
tf.flags.DEFINE_integer("action_space", 3, "Dimensionality of action space")
tf.flags.DEFINE_integer("state_space", 1, "Dimensionality of state space")

# posterior
tf.flags.DEFINE_float("prior_precision", 0.5, "Prior precision (1/var)")
tf.flags.DEFINE_float("noise_precision", 0.1, "Noise precision (1/var)")
tf.flags.DEFINE_float("noise_precmax", 20., "Maximum noise precision (1/var)")
tf.flags.DEFINE_integer("noise_Ndrop", 1, "Increase noise precision every N steps")
tf.flags.DEFINE_float("noise_precstep", 1.001, "Step of noise precision s*=ds")

tf.flags.DEFINE_integer("split_N", 20, "Increase split ratio every N steps")
tf.flags.DEFINE_float("split_ratio", 0.7, "Initial split ratio for conditioning")
tf.flags.DEFINE_float("split_ratio_max", 0.7, "Maximum split ratio for conditioning")
tf.flags.DEFINE_integer("update_freq_post", 1, "Update frequency of posterior and sampling of new policy")

# exploration
tf.flags.DEFINE_float("eps_initial", 0., "Initial value for epsilon-greedy")
tf.flags.DEFINE_float("eps_final", 0., "Final value for epsilon-greedy")
tf.flags.DEFINE_float("eps_step", 0.9995, "Multiplicative step for epsilon-greedy")

# target
tf.flags.DEFINE_float("tau", 0.01, "Update speed of target network")
tf.flags.DEFINE_integer("update_freq_target", 1, "Update frequency of target network")

# loss
tf.flags.DEFINE_float("learning_rate", 1e-2, "Initial learning rate") # X
tf.flags.DEFINE_float("lr_drop", 1.0003, "Drop of learning rate per episode")
tf.flags.DEFINE_float("grad_clip", 1e4, "Absolute value to clip gradients")
tf.flags.DEFINE_float("huber_d", 1e0, "Switch point from quadratic to linear")
tf.flags.DEFINE_float("regularizer", 1e-3, "Regularization parameter") # X

# reward
tf.flags.DEFINE_float("rew_norm", 1e0, "Normalization factor for reward")
tf.flags.DEFINE_bool("rew_log", True, "If Log of reward is taken")

# memory
tf.flags.DEFINE_integer("replay_memory_size", 100, "Size of replay memory")
tf.flags.DEFINE_integer("iter_amax", 1, "Number of iterations performed to determine amax")
tf.flags.DEFINE_integer("save_frequency", 500, "Store images every N-th episode")

#
tf.flags.DEFINE_integer("random_seed", 1234, "Random seed for numpy and tensorflow")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.38)

from QNetwork import QNetwork

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
rt_dir = 'figures/'+ time.strftime('%H-%M-%d_%m-%y')+ '/rt/'
if not os.path.exists(rt_dir):
    os.makedirs(rt_dir)

r0_dir = 'figures/'+ time.strftime('%H-%M-%d_%m-%y')+ '/r0/'
if not os.path.exists(r0_dir):
    os.makedirs(r0_dir)

basis_fcn_dir = 'figures/'+ time.strftime('%H-%M-%d_%m-%y')+ '/basis_fcn/'
if not os.path.exists(basis_fcn_dir):
    os.makedirs(basis_fcn_dir)

reward_dir = 'figures/'+ time.strftime('%H-%M-%d_%m-%y')+ '/'
if not os.path.exists(reward_dir):
    os.makedirs(reward_dir)

# initialize replay memory and model
fullbuffer = replay_buffer(FLAGS.replay_memory_size) # large buffer to store all experience
tempbuffer = replay_buffer(FLAGS.L_episode) # buffer for episode
log.info('Build Tensorflow Graph')

# initialize environment
env = bandit_environment(FLAGS.action_space)

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
    loss0Buffer = 0.
    loss1Buffer = 0.
    loss2Buffer = 0.

    lossregBuffer = 0.

    # report mean reward per episode
    reward_write_to_file = []
    regret_write_to_file = []

    # -----------------------------------------------------------------------------------
    # loop episodes
    print("Episodes...")
    log.info('Loop over episodes')
    for episode in range(FLAGS.N_episodes):

        # count reward
        reward_agent = []
        reward_opt = []
        reward_rand = []

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
                Qval = sess.run([QNet.Qout], feed_dict={QNet.state: state.reshape(-1,FLAGS.state_space)})
                action = eGreedyAction(Qval, eps)
                next_state, reward, done, rew_max, rew_rand = env._step(action)

                # store experience in memory
                new_experience = [state, action, reward, next_state, done]
                tempbuffer.add(new_experience)


                if step == 0 and n == 0 and episode % FLAGS.save_frequency == 0:

                    # plot w* phi
                    env_state = np.linspace(0, 1, 100)
                    env_phase = env.phase
                    env_psi = env._psi(env_state, env_phase)
                    env_theta = env.theta

                    w0_bar, L0, Sigma_e, phi = sess.run([QNet.w0_bar, QNet.L0, QNet.Sigma_e, QNet.phi],
                                                        feed_dict={QNet.state: env_state.reshape(-1, 1),
                                                                   QNet.nprec: noise_precision})

                    fig, ax = plt.subplots(ncols=FLAGS.action_space, figsize=[20, 5])

                    for act in range(FLAGS.action_space):
                        env_r = env_theta[act] * env_psi[:, act]

                        Q_r = np.dot(phi[:, :, act], w0_bar).reshape(-1)
                        dQ_r = np.sqrt(np.einsum('bi,ij,bj->b', phi[:, :, act], np.linalg.inv(L0), phi[:, :, act])+ Sigma_e)

                        Q0 = np.dot(phi[:, :, act], w0_bar)

                        if FLAGS.action_space > 1:
                            ax[act].plot(env_state, env_r, 'r')
                            ax[act].plot(env_state, Q_r, 'b--')
                            ax[act].plot(env_state, Q0, 'b')
                            ax[act].fill_between(env_state, Q_r - 1.96 * dQ_r, Q_r + 1.96 * dQ_r, alpha=0.5)
                            ax[act].set_xlim([0., 1.])
                            ax[act].set_ylim([-7, 7])
                            ax[act].set_xlabel('State')
                            ax[act].set_ylabel('Reward')
                        else:
                            ax.plot(env_state, env_r, 'r')
                            ax.plot(env_state, Q_r, 'b')
                            ax.plot(env_state, Q0, 'b--')
                            ax.fill_between(env_state, Q_r - 1.96 * dQ_r, Q_r + 1.96 * dQ_r, alpha=0.5)

                    plt.rc('font', size=16)
                    plt.tight_layout()
                    plt.savefig(rt_dir + 'Epoch_' + str(episode) + '_step_' + str(step) + '_Reward')
                    plt.close()



                # update posterior
                # TODO: could speed up by iteratively adding
                if (step+1) % FLAGS.update_freq_post == 0:
                    reward_train = np.zeros([step+1, ])
                    state_train = np.zeros([step+1, FLAGS.state_space])
                    next_state_train = np.zeros([step+1, FLAGS.state_space])
                    action_train = np.zeros([step+1, ])
                    done_train = np.zeros([step+1])

                    # fill arrays
                    for k, experience in enumerate(tempbuffer.buffer):
                        # [s, a, r, s', a*, d]
                        state_train[k] = experience[0]
                        action_train[k] = experience[1]
                        reward_train[k] = experience[2]
                        next_state_train[k] = experience[3]
                        done_train[k] = experience[4]

                    # update
                    _, wt_bar, Lt_inv, phi_next, phi_taken = sess.run([QNet.sample_post, QNet.wt_bar, QNet.Lt_inv, QNet.context_phi_next, QNet.context_phi_taken],
                             feed_dict={QNet.context_state: state_train,
                                        QNet.context_action: action_train,
                                        QNet.context_reward: reward_train,
                                        QNet.context_state_next: next_state_train,
                                        QNet.context_done: done_train,
                                        QNet.nprec: noise_precision})

                    # plot
                    if episode % FLAGS.save_frequency == 0 and n == 0:

                        # plot w* phi
                        env_state = np.linspace(0, 1, 100)
                        env_phase = env.phase
                        env_psi = env._psi(env_state, env_phase)
                        env_theta = env.theta

                        phi, w0_bar, Sigma_e = sess.run([QNet.phi, QNet.w0_bar, QNet.Sigma_e],
                                                        feed_dict={QNet.state: env_state.reshape(-1,1),
                                                                   QNet.nprec: noise_precision})

                        fig, ax = plt.subplots(ncols=FLAGS.action_space, figsize=[20, 5])

                        for act in range(FLAGS.action_space):
                            env_r = env_theta[act] * env_psi[:, act]

                            Q_r = np.dot(phi[:, :, act], wt_bar).reshape(-1)
                            dQ_r = np.sqrt(np.einsum('bi,ij,bj->b', phi[:, :, act], Lt_inv, phi[:, :, act])+ Sigma_e) # stdv

                            Q0 = np.dot(phi[:, :, act], w0_bar)

                            delta = np.where(action_train == act) # to visualize which action network took

                            if FLAGS.action_space > 1:
                                ax[act].plot(env_state, env_r, 'r')
                                ax[act].plot(env_state, Q_r, 'b--')
                                ax[act].plot(env_state, Q0, 'b')
                                ax[act].scatter(state_train[delta], reward_train[delta], marker='x', color='r', s=60)
                                ax[act].fill_between(env_state, Q_r - 1.96*dQ_r, Q_r + 1.96*dQ_r, alpha=0.5)
                                ax[act].set_xlim([0., 1.])
                                ax[act].set_ylim([-7, 7])
                                ax[act].set_xlabel('State')
                                ax[act].set_ylabel('Reward')
                            else:
                                ax.plot(env_state, env_r, 'r')
                                ax.plot(env_state, Q_r, 'b')
                                ax.plot(env_state, Q0, 'b--')
                                ax.scatter(state_train[delta], reward_train[delta], marker='x', color='r')
                                ax.fill_between(env_state, Q_r - 1.96*dQ_r, Q_r + 1.96*dQ_r, alpha=0.5)

                        plt.rc('font', size=16)
                        plt.tight_layout()
                        plt.savefig(rt_dir + 'Epoch_' + str(episode)+ '_step_'+ str(step+1) + '_Reward')
                        plt.close()


                # update state, and counters
                state = next_state.copy()
                global_index += 1
                step += 1

                # count reward
                reward_agent.append(reward)
                reward_opt.append(rew_max)
                reward_rand.append(rew_rand)

                # -----------------------------------------------------------------------

            # append episode buffer to large buffer
            fullbuffer.add(tempbuffer.buffer)

        # append reward
        reward_write_to_file.append(np.sum(np.array(reward_agent))/ FLAGS.N_tasks)

        regret = (np.sum(np.asarray(reward_opt)) - np.sum(np.asarray(reward_agent))) / \
                 (np.sum(np.asarray(reward_opt)) - np.sum(np.asarray(reward_rand)))

        regret_write_to_file.append(regret) # no need to divide by #tasks

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
            split = np.int(split_ratio* FLAGS.L_episode* np.random.rand())

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

            # update model
            grads, loss0, loss1, loss2, loss_reg, loss = sess.run([QNet.gradients, QNet.loss0, QNet.loss1,
                                                                          QNet.loss2, QNet.loss_reg, QNet.loss],
                                                feed_dict={QNet.context_state: state_train,
                                                           QNet.context_action: action_train,
                                                           QNet.context_reward: reward_train,
                                                           QNet.context_state_next: next_state_train,
                                                           QNet.context_done: done_train,
                                                           QNet.state: state_valid,
                                                           QNet.action: action_valid,
                                                           QNet.reward: reward_valid,
                                                           QNet.state_next: next_state_valid,
                                                           QNet.done: done_valid,
                                                           QNet.lr_placeholder: learning_rate,
                                                           QNet.nprec: noise_precision})


            for idx, grad in enumerate(grads): # grad[0] is gradient and grad[1] the variable itself
                gradBuffer[idx] += (grad[0]/ batch_size)

            lossBuffer += loss
            loss0Buffer += loss0
            loss1Buffer += loss1
            loss2Buffer += loss2
            lossregBuffer += loss_reg

        # update summary
        feed_dict= dictionary = dict(zip(QNet.gradient_holders, gradBuffer))
        feed_dict.update({QNet.lr_placeholder: learning_rate})
        _, summaries_gradvar = sess.run([QNet.updateModel, QNet.summaries_gradvar], feed_dict=feed_dict)

        # reduce summary size
        if episode % 10 == 0:
            loss_summary = tf.Summary(value=[tf.Summary.Value(tag='Loss/Loss', simple_value=(lossBuffer / batch_size))])
            loss0_summary = tf.Summary(value=[tf.Summary.Value(tag='Loss/TD_Loss', simple_value=(loss0Buffer / batch_size))])
            loss1_summary = tf.Summary(value=[tf.Summary.Value(tag='Loss/TDW_Loss', simple_value=(loss1Buffer/ batch_size))])
            loss2_summary = tf.Summary(value=[tf.Summary.Value(tag='Loss/Sig_Loss', simple_value=(loss2Buffer / batch_size))])
            lossreg_summary = tf.Summary(value=[tf.Summary.Value(tag='Loss/Regularization_Loss', simple_value=(lossregBuffer / batch_size))])
            reward_summary = tf.Summary(value=[tf.Summary.Value(tag='Performance/Episodic Reward', simple_value=np.sum(np.array(reward_agent)))])
            regret_summary = tf.Summary(value=[tf.Summary.Value(tag='Performance/Episodic Regret', simple_value=regret)])
            learning_rate_summary = tf.Summary(value=[tf.Summary.Value(tag='Parameter/Learning rate', simple_value=learning_rate)])

            summary_writer.add_summary(loss_summary, episode)
            summary_writer.add_summary(loss0_summary, episode)
            summary_writer.add_summary(loss1_summary, episode)
            summary_writer.add_summary(loss2_summary, episode)
            summary_writer.add_summary(lossreg_summary, episode)
            summary_writer.add_summary(reward_summary, episode)
            summary_writer.add_summary(regret_summary, episode)
            summary_writer.add_summary(summaries_gradvar, episode)
            summary_writer.add_summary(learning_rate_summary, episode)

            summary_writer.flush()

        # reset buffers
        for idx in range(len(gradBuffer)):
            gradBuffer[idx] *= 0

        lossBuffer *= 0.
        loss0Buffer *= 0.
        loss1Buffer *= 0.
        loss2Buffer *= 0.
        lossregBuffer *= 0.

        # increase the batch size after the first episode. Would allow N_tasks < batch_size due to buffer
        if episode < 2:
            batch_size *= 2

        # learning rate schedule
        if learning_rate > 5e-6:
            learning_rate /= FLAGS.lr_drop

        if noise_precision < FLAGS.noise_precmax and episode % FLAGS.noise_Ndrop == 0:
            noise_precision *= FLAGS.noise_precstep

        if episode % FLAGS.split_N == 0 and episode > 0:
            split_ratio = np.min([split_ratio+ 0.01, FLAGS.split_ratio_max])

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
            log.info('Episode %3.d with R %3.d', episode, np.sum(reward_agent))

            print('Reward in Episode ' + str(episode) + ':   ' + str(np.sum(reward_agent)))
            print('Learning_rate: ' + str(np.round(learning_rate, 5)) + ', Nprec: ' + str(noise_precision))

            # plot w* phi
            env_state = np.linspace(0, 1, 100)
            env_psi = env._psi(env_state, np.pi/2.* np.ones(FLAGS.action_space))
            env_mu = env.mu

            w0_bar, L0, Sigma_e, phi = sess.run([QNet.w0_bar, QNet.L0, QNet.Sigma_e, QNet.phi],
                                                feed_dict={QNet.state: env_state.reshape(-1,1),
                                                           QNet.nprec: noise_precision})

            fig, ax = plt.subplots(figsize=[10,5])
            color = iter(cm.rainbow(np.linspace(0, 1, phi.shape[1])))
            for bf in range(phi.shape[1]):
                ax.plot(phi[:, bf, 0], c=next(color))

            plt.savefig(basis_fcn_dir + 'Epoch_' + str(episode))
            plt.close()


            # plot prior
            fig, ax = plt.subplots(ncols=FLAGS.action_space, figsize=[20,5])

            for act in range(FLAGS.action_space):
                env_r = env_mu[act] * env_psi[:, act]

                Q_r = np.dot(phi[:, :, act], w0_bar[:, 0])
                dQ_r = np.sqrt(np.einsum('bi,ij,bj->b', phi[:, :, act], np.linalg.inv(L0), phi[:, :, act])+ Sigma_e)

                if FLAGS.action_space > 1:
                    ax[act].plot(env_state, env_r, 'r')
                    ax[act].plot(env_state, Q_r, 'b')
                    ax[act].fill_between(env_state, Q_r- 1.96*dQ_r, Q_r+ 1.96*dQ_r, alpha=0.5)
                    ax[act].set_xlim([0., 1.])
                    ax[act].set_ylim([-7, 7])
                    ax[act].set_xlabel('State')
                    ax[act].set_ylabel('Reward')
                else:
                    ax.plot(env_state, env_r, 'r')
                    ax.plot(env_state, Q_r, 'b')
                    ax.fill_between(env_state, Q_r - 1.96*dQ_r, Q_r + 1.96*dQ_r, alpha=0.5)

            plt.rc('font', size=16)
            plt.tight_layout()
            plt.savefig(r0_dir+'Epoch_'+str(episode)+'_Reward')
            plt.close()

        if episode % 5000 == 0:
            # evaluation =================================================================
            print('Evaluation ===================')
            # cumulative regret
            cumulative_regret = []

            # simple regret
            simple_regret = []

            for test_ep in range(500):
                # new environment
                env._sample_env()
                state = env._sample_state().copy()

                # reset hidden states
                sess.run([QNet.sample_prior])

                # reset buffer
                tempbuffer.reset()

                # initialize
                step = 0

                # inner loop (cumulative regret)
                reward_agent = 0
                reward_max = 0
                reward_rand = 0

                while step < FLAGS.L_episode:
                    # take a step
                    Qval = sess.run([QNet.Qout], feed_dict={QNet.state: state.reshape(-1, FLAGS.state_space)})
                    action = eGreedyAction(Qval, eps)
                    next_state, rew, done, rew_max, rew_rand = env._step(action)

                    # store experience in memory
                    new_experience = [state, action, rew, next_state, done]
                    tempbuffer.add(new_experience)

                    # iterate
                    state = next_state.copy()
                    step += 1

                    # rewards
                    reward_agent += rew
                    reward_max += rew_max
                    reward_rand += rew_rand

                    # update posterior
                    if (step) % FLAGS.update_freq_post == 0:
                        reward_train = np.zeros([step, ])
                        state_train = np.zeros([step, FLAGS.state_space])
                        next_state_train = np.zeros([step, FLAGS.state_space])
                        action_train = np.zeros([step, ])
                        done_train = np.zeros([step])

                        # fill arrays
                        for k, experience in enumerate(tempbuffer.buffer):
                            # [s, a, r, s', a*, d]
                            state_train[k] = experience[0]
                            action_train[k] = experience[1]
                            reward_train[k] = experience[2]
                            next_state_train[k] = experience[3]
                            done_train[k] = experience[4]

                        # update
                        _ = sess.run(QNet.sample_post,
                            feed_dict={QNet.context_state: state_train,
                                       QNet.context_action: action_train,
                                       QNet.context_reward: reward_train,
                                       QNet.context_state_next: next_state_train,
                                       QNet.context_done: done_train,
                                       QNet.nprec: noise_precision})

                #
                cumulative_regret.append((reward_max - reward_agent) / (reward_max - reward_rand))

                # no updates to hidden state (simple regret)
                reward_agent = 0
                reward_max = 0
                reward_rand = 0

                for _ in range(FLAGS.L_episode):
                    # take a step
                    Qval = sess.run([QNet.Qout], feed_dict={QNet.state: state.reshape(-1, FLAGS.state_space)})
                    action = eGreedyAction(Qval, eps)
                    next_state, rew, done, rew_max, rew_rand = env._step(action)

                    # iterate
                    state = next_state.copy()

                    # rewards
                    reward_agent += rew
                    reward_max += rew_max
                    reward_rand += rew_rand

                simple_regret.append((reward_max - reward_agent) / (reward_max - reward_rand))

            print('Mean Cumulative Regret: {}'.format(np.mean(np.asarray(cumulative_regret))))
            print('Mean Simple Regret: {}'.format(np.mean(np.asarray(simple_regret))))

            file = open(reward_dir + 'test_regret_per_episode', 'a')
            file.write('Cumulative Regret\n')
            file.write(
                '{:3.4f}% +- {:2.4f}%\n'.format(np.mean(np.asarray(cumulative_regret)), np.std(np.asarray(cumulative_regret))))
            file.write('Simple Regret\n')
            file.write('{:3.4f}% +- {:2.4f}%\n'.format(np.mean(np.asarray(simple_regret)), np.std(np.asarray(simple_regret))))
            file.close()

    # write reward to file
    df = pd.DataFrame(reward_write_to_file)
    df.to_csv(reward_dir+'reward_per_episode', index=False)

    df = pd.DataFrame(regret_write_to_file)
    df.to_csv(reward_dir + 'regret_per_episode', index=False)

    # reset buffers
    try:
        fullbuffer.reset()
    except:
        print('')
    try:
        tempbuffer.reset()
    except:
        print('')
