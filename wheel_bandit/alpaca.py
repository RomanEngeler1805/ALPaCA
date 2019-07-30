'''
Class structure for ALPaCA meta-RL code
'''

import numpy as np
import tensorflow as tf
import os
import time
import logging
import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.30) #0.13, 0.3

# General Hyperparameters
tf.flags.DEFINE_integer("batch_size", 2, "Batch size for training")
tf.flags.DEFINE_integer("action_space", 5, "Dimensionality of action space")
tf.flags.DEFINE_integer("state_space", 2, "Dimensionality of state space")
tf.flags.DEFINE_integer("hidden_space", 128, "Dimensionality of hidden space")
tf.flags.DEFINE_integer("latent_space", 64, "Dimensionality of latent space")
tf.flags.DEFINE_float("gamma", 0., "Discount factor")
tf.flags.DEFINE_float("learning_rate", 5e-3, "Initial learning rate")
tf.flags.DEFINE_float("lr_drop", 1.00015, "Drop of learning rate per episode")
tf.flags.DEFINE_float("prior_precision", 0.5, "Prior precision (1/var)")

tf.flags.DEFINE_float("noise_precision", 0.1, "Noise precision (1/var)")
tf.flags.DEFINE_float("noise_precmax", 20, "Maximum noise precision (1/var)")
tf.flags.DEFINE_integer("noise_Ndrop", 30, "Increase noise precision every N steps")
tf.flags.DEFINE_float("noise_precstep", 1.001, "Step of noise precision s*=ds")

tf.flags.DEFINE_integer("split_N", 30, "Increase split ratio every N steps")
tf.flags.DEFINE_float("split_ratio", 0.0, "Initial split ratio for conditioning")

tf.flags.DEFINE_integer("kl_freq", 100, "Update kl divergence comparison")
tf.flags.DEFINE_float("kl_lambda", 10., "Weight for Kl divergence in loss")

tf.flags.DEFINE_integer("N_episodes", 4000, "Number of episodes")
tf.flags.DEFINE_integer("N_tasks", 2, "Number of tasks")
tf.flags.DEFINE_integer("L_episode", 50, "Length of episodes")

tf.flags.DEFINE_float("tau", 1., "Update speed of target network")
tf.flags.DEFINE_integer("update_freq_target", 20, "Update frequency of target network")

tf.flags.DEFINE_integer("replay_memory_size", 100, "Size of replay memory")
tf.flags.DEFINE_integer("update_freq", 1, "Update frequency of posterior and sampling of new policy")
tf.flags.DEFINE_integer("iter_amax", 1, "Number of iterations performed to determine amax")
tf.flags.DEFINE_integer("save_frequency", 400, "Store images every N-th episode")
tf.flags.DEFINE_float("regularizer", 0.01, "Regularization parameter")
tf.flags.DEFINE_string('non_linearity', 'relu', 'Non-linearity used in encoder')

tf.flags.DEFINE_integer("random_seed", 1805, "Random seed for numpy and tensorflow")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)

np.random.seed(FLAGS.random_seed)
tf.set_random_seed(FLAGS.random_seed)

from wheel_bandit_environment import wheel_bandit_environment
from QNetwork import QNetwork
from plot_Value_fcn import plot_Value_fcn

sys.path.insert(0, './..')
from replay_buffer import replay_buffer


def eGreedyAction(x, epsilon=0.9):
    ''' select next action according to epsilon-greedy algorithm '''
    if np.random.rand() >= epsilon:
        action = np.argmax(x)
    else:
        action = np.random.randint(FLAGS.action_space)

    return action

def quadrant(pos, delta):
    quad = 0

    if np.linalg.norm(pos) < delta:
        quad = 0
    else:
        if pos[0] > 0 and pos[1] > 0:
            quad = 1
        elif pos[0] < 0 and pos[1] > 0:
            quad = 2
        elif pos[0] < 0 and pos[1] < 0:
            quad = 3
        elif pos[0] > 0 and pos[1] < 0:
            quad = 4
        else:
            return -1

    return quad


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
histo_dir = 'figures/'+ time.strftime('%H-%M-%d_%m-%y')+ '/histo/'
if not os.path.exists(histo_dir):
    os.makedirs(histo_dir)

reward_dir = 'figures/'+ time.strftime('%H-%M-%d_%m-%y')+ '/'
if not os.path.exists(reward_dir):
    os.makedirs(reward_dir)

# initialize replay memory and model
fullbuffer = replay_buffer(FLAGS.replay_memory_size) # large buffer to store all experience
tempbuffer = replay_buffer(FLAGS.L_episode) # buffer for episode
rewardbuffer = replay_buffer(FLAGS.L_episode) # buffer for episode
log.info('Build Tensorflow Graph')

# initialize environment
env = wheel_bandit_environment(FLAGS.action_space, FLAGS.random_seed)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    QNet = QNetwork(FLAGS, scope='QNetwork')  # neural network
    Qtarget = QNetwork(FLAGS, scope='TargetNetwork')

    # session
    init = tf.global_variables_initializer()
    sess.run(init)

    # DKL with old values (limit rate of change)
    w0_bar_old = np.zeros([2, FLAGS.latent_space, 1])
    L0_asym_old = np.zeros([2, FLAGS.latent_space])

    w0_bar_old[0], L0_asym_old[0] = sess.run([QNet.w0_bar, QNet.L0_asym])
    w0_bar_old[1] = w0_bar_old[0].copy()
    L0_asym_old[1] = L0_asym_old[0].copy()

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
        r_star = []
        r_uni = []

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

            reward_train = []
            state_train = []
            next_state_train = []
            action_train = []
            done_train = []

            while step < FLAGS.L_episode:

                # max reward (neglect stdv)
                r_star.append(np.max(env.mu[env._mu_idx(env.state)]))

                # uniform reward
                if np.linalg.norm(env.state) <= env.delta:
                    r_uni.append(1./5.* env.mu[0]+ 4./5.* env.mu[1])
                else:
                    r_uni.append(1./5.* env.mu[0]+ 3./5.* env.mu[1]+ 1./5.* env.mu[2])

                # take a step
                Qval = sess.run([QNet.Qout], feed_dict={QNet.state: state.reshape(-1,FLAGS.state_space)})
                action = eGreedyAction(Qval, eps)
                next_state, reward, done = env._step(action)

                # store experience in memory
                new_experience = [state, action, reward, next_state, done]
                tempbuffer.add(new_experience)

                reward_train.append(reward)
                state_train.append(state)
                next_state_train.append(next_state)
                action_train.append(action)
                done_train.append(done)

                # actual reward
                rw.append(reward)

                # update posterior
                if (step+1) % FLAGS.update_freq == 0 and (step+1) <= np.int(split_ratio* FLAGS.L_episode):
                    # update
                    _, wt_bar, Lt_inv, phi_next, phi_taken = sess.run([QNet.sample_post, QNet.wt_bar, QNet.Lt_inv, QNet.context_phi_next, QNet.context_phi_taken],
                             feed_dict={QNet.context_state: np.asarray(state_train),
                                        QNet.context_action: np.asarray(action_train),
                                        QNet.context_reward: np.asarray(reward_train),
                                        QNet.context_state_next: np.asarray(next_state_train),
                                        QNet.context_done: np.asarray(done_train).reshape(-1,1),
                                        QNet.nprec: noise_precision})

                # update state, and counters
                state = next_state.copy()
                global_index += 1
                step += 1

                # -----------------------------------------------------------------------
            # append episode buffer to large buffer
            fullbuffer.add(tempbuffer.buffer)

        # append reward
        reward_episode.append(np.sum(np.array(rw))/(FLAGS.N_tasks* FLAGS.L_episode))

        # learning rate schedule
        if learning_rate > 5e-5:
            learning_rate /= FLAGS.lr_drop

        if noise_precision < FLAGS.noise_precmax and episode % FLAGS.noise_Ndrop == 0:
            noise_precision *= FLAGS.noise_precstep

        if episode % FLAGS.split_N == 0 and episode > 0:
            split_ratio = np.min([split_ratio+ 0.01, 0.9])

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

            # select amax from online network
            amax_online = sess.run(QNet.max_action,
                feed_dict={QNet.context_state: state_train, QNet.context_action: action_train,
                           QNet.context_reward: reward_train,
                           QNet.context_state_next: next_state_train,
                           QNet.state: state_valid, QNet.state_next: next_state_valid,
                           QNet.nprec: noise_precision})

            # evaluate target model
            Qmax_target = sess.run(Qtarget.Qmax,
                feed_dict={Qtarget.context_state: state_train, Qtarget.context_action: action_train,
                           Qtarget.context_reward: reward_train, Qtarget.context_state_next: next_state_train,
                           Qtarget.state: state_valid, Qtarget.state_next: next_state_valid,
                           Qtarget.amax_online: amax_online,
                           Qtarget.nprec: noise_precision})

            # update model
            grads, loss0, loss1, loss2, loss3, loss_reg, loss, summaries_encodinglayer = sess.run(
                [QNet.gradients, QNet.loss0, QNet.loss1, QNet.loss2, QNet.loss3, QNet.loss_reg, QNet.loss, QNet.summaries_encodinglayer],
                feed_dict={QNet.context_state: state_train, QNet.context_action: action_train,
                           QNet.context_reward: reward_train, QNet.context_state_next: next_state_train,
                           QNet.state: state_valid, QNet.action: action_valid,
                           QNet.reward: reward_valid, QNet.state_next: next_state_valid,
                           QNet.done: done_valid, QNet.Qmax_target: Qmax_target,
                           QNet.amax_online: amax_online,
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

            #summaries_var = sess.run(Qtarget.summaries_var)

            loss_summary = tf.Summary(value=[tf.Summary.Value(tag='Loss', simple_value=(lossBuffer / batch_size))])
            loss0_summary = tf.Summary(value=[tf.Summary.Value(tag='TD_Loss', simple_value=(loss0Buffer / batch_size))])
            loss1_summary = tf.Summary(value=[tf.Summary.Value(tag='TDW_Loss', simple_value=(loss1Buffer/ batch_size))])
            loss2_summary = tf.Summary(value=[tf.Summary.Value(tag='Sig_Loss', simple_value=(loss2Buffer / batch_size))])
            loss3_summary = tf.Summary(value=[tf.Summary.Value(tag='KL_Loss', simple_value=(loss3Buffer / batch_size))])
            lossreg_summary = tf.Summary(value=[tf.Summary.Value(tag='Regularization_Loss', simple_value=(lossregBuffer / batch_size))])
            reward_norm_summary = tf.Summary(value=[tf.Summary.Value(tag='Episodic Reward (Normalized)', simple_value=np.sum(np.array(rw)) / np.sum(np.array(r_star)))])
            regret_norm_summary = tf.Summary(value=[tf.Summary.Value(tag='Episodic Regret (normalized)',
                            simple_value=(np.sum(np.array(r_star))- np.sum(np.array(rw)))/ (np.sum(np.array(r_star)) - np.sum(np.array(r_uni))))])
            learning_rate_summary = tf.Summary(value=[tf.Summary.Value(tag='Learning rate', simple_value=learning_rate)])

            summary_writer.add_summary(loss_summary, episode)
            summary_writer.add_summary(loss0_summary, episode)
            summary_writer.add_summary(loss1_summary, episode)
            summary_writer.add_summary(loss2_summary, episode)
            summary_writer.add_summary(loss3_summary, episode)
            summary_writer.add_summary(lossreg_summary, episode)
            summary_writer.add_summary(reward_norm_summary, episode)
            summary_writer.add_summary(regret_norm_summary, episode)
            #summary_writer.add_summary(summaries_var, episode)
            summary_writer.add_summary(summaries_gradvar, episode)
            summary_writer.add_summary(summaries_encodinglayer, episode)
            summary_writer.add_summary(learning_rate_summary, episode)

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
        if episode < 3:
            batch_size *= 2

        # kl divergence updates
        if episode % FLAGS.kl_freq == 0:
            w0_bar_old[0] = w0_bar_old[1]
            L0_asym_old[0] = L0_asym_old[1]

            w0_bar_old[1], L0_asym_old[1] = sess.run([QNet.w0_bar, QNet.L0_asym])

        # ===============================================================
        # update target network
        if episode % FLAGS.update_freq_target == 0:
            #
            vars_modelQ = sess.run(QNet.tvars)

            #
            feed_dict = dictionary = dict(zip(Qtarget.variable_holders, vars_modelQ))
            feed_dict.update({Qtarget.tau: FLAGS.tau})

            #
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
            print('Learning_rate: '+ str(np.round(learning_rate,4))+ ', Nprec: '+ str(np.round(noise_precision,4))+ ', Split ratio: '+ str(np.round(split_ratio, 2)))

            log.info('Episode %3.d with time per episode %5.2f', episode, (time.time()- start))

            # evaluation of regret to compare against bayesian bandit showdown paper ===================================
            L_valid = 50
            n_sets = 10
            n_shuffle = 12

            if episode == 0:
                # fill context array
                reward_context = np.zeros([n_sets, L_valid, FLAGS.state_space*2]) #+1 since next_state
                for ns in range(n_sets):
                    for rc in range(L_valid):
                        reward_context[ns, rc, :2] = env._sample_state() # cartesian
                        reward_context[ns, rc, 2:] = env.state # cylindrical

            # loop over exploration parameter
            for delta_eval in [0.5, 0.7, 0.9, 0.95, 0.99]:
                env.delta = delta_eval

                cum_regr_QN = np.zeros([n_sets, n_shuffle])
                cum_regr_UF = np.zeros([n_sets, n_shuffle])

                cum_rew_QN = np.zeros([n_sets, n_shuffle])
                cum_rew_UF = np.zeros([n_sets, n_shuffle])
                cum_rew_ST = np.zeros([n_sets, n_shuffle])

                # ----------------------------------------------------------------------------------------------
                dir = histo_dir + '/' + str(np.int(100 * delta_eval)) + '/'
                if not os.path.exists(dir):
                    os.makedirs(dir)
                path = dir + 'Epoch_' + str(episode)
                plot_Value_fcn(path, env, sess, QNet, noise_precision)

                # -----------------------------------------------------------------------------------------------

                for ns in range(n_sets):

                    set = reward_context[ns]

                    for sh in range(n_shuffle): # shuffling
                        np.random.shuffle(set)

                        # data buffer
                        rewardbuffer.reset()

                        # store expected rewards
                        rw = []
                        r_star = []
                        r_uni = []

                        # loop steps
                        step = 0

                        # resample state
                        state = set[step, :2] # cartesian

                        # sample w from prior
                        sess.run([QNet.sample_prior])

                        while step < L_valid:
                            # --------------------------------------
                            # get index of mean value for all actions (before state internal env state is updated)
                            mu_idx = env._mu_idx(set[step, 2:]) # cylindrical
                            # get mean reward for each action
                            mu = env.mu[mu_idx]

                            # expected max reward
                            r_star.append(np.max(mu))

                            # expected uniform reward
                            r_uni.append(np.mean(mu))

                            # take a step
                            Qval = sess.run([QNet.Qout], feed_dict={QNet.state: state.reshape(-1, FLAGS.state_space)})
                            action = eGreedyAction(Qval, eps)

                            # expected model reward
                            rw.append(mu[action])

                            next_state, reward, done = [set[(step+1)%L_valid, :2], mu[action], 0] # cartesian

                            # store experience in memory
                            new_experience = [state, action, reward, next_state, done]
                            rewardbuffer.add(new_experience)

                            # update
                            state = next_state.copy()
                            step+= 1

                            # update posterior
                            if (step + 1) % FLAGS.update_freq == 0 and (step + 1) <= np.int(split_ratio * FLAGS.L_episode):
                                reward_train = np.zeros([step + 1, ])
                                state_train = np.zeros([step + 1, FLAGS.state_space])
                                next_state_train = np.zeros([step + 1, FLAGS.state_space])
                                action_train = np.zeros([step + 1, ])
                                done_train = np.zeros([step + 1, 1])

                                # fill arrays
                                for k, experience in enumerate(rewardbuffer.buffer):
                                    # [s, a, r, s', a*, d]
                                    state_train[k] = experience[0]
                                    action_train[k] = experience[1]
                                    reward_train[k] = experience[2]
                                    next_state_train[k] = experience[3]
                                    done_train[k] = experience[4]

                                # update
                                _ = sess.run(QNet.sample_post,
                                    feed_dict={QNet.context_state: state_train, QNet.context_action: action_train,
                                               QNet.context_reward: reward_train, QNet.context_state_next: next_state_train,
                                               QNet.context_done: done_train,
                                               QNet.nprec: noise_precision})

                        if np.sum(np.array(r_star)) < np.sum(np.array(rw)):
                            print('ERROR !!!')

                        cum_rew_QN[ns, sh] = np.sum(np.array(rw))
                        cum_rew_UF[ns, sh] = np.sum(np.array(r_uni))
                        cum_rew_ST[ns, sh] = np.sum(np.array(r_star))

                        cum_regr_QN[ns, sh] = np.sum(np.array(r_star)) - np.sum(np.array(rw))
                        cum_regr_UF[ns, sh] = np.sum(np.array(r_star)) - np.sum(np.array(r_uni))

                    # plot with trajectory
                    path = dir + 'Epoch_' + str(episode)+ '_observations'
                    plot_Value_fcn(path, env, sess, QNet, noise_precision, rewardbuffer.buffer)

                # summaries
                val_rew_summary = tf.Summary(value=[tf.Summary.Value(tag='Validation Reward normalized (delta '+ str(delta_eval)+ ')',
                                                                     simple_value=np.mean(cum_rew_QN)/ np.mean(cum_rew_ST))])

                summary_writer.add_summary(val_rew_summary, episode)

                regret_valid = np.mean(cum_regr_QN, axis=1) / np.mean(cum_regr_UF, axis=1)* 100
                regret_valid = np.mean(regret_valid)

                tag_name = 'Validation Cumulative Regret (delta '+ str(delta_eval)+ ')'

                regret_norm_summary = tf.Summary(value=[tf.Summary.Value(tag=tag_name, simple_value=regret_valid)])
                summary_writer.add_summary(regret_norm_summary, episode)

            summary_writer.flush()


    # write reward to file
    df = pd.DataFrame(reward_episode)
    df.to_csv(reward_dir+'reward_per_episode', index=False)

    # reset buffers
    fullbuffer.reset()
    tempbuffer.reset()
