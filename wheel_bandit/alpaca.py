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
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.20)

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
tf.flags.DEFINE_float("noise_precision", 0.01, "Noise precision (1/var)")
tf.flags.DEFINE_float("noise_precmax", 20, "Maximum noise precision (1/var)")
tf.flags.DEFINE_integer("noise_Ndrop", 30, "Increase noise precision every N steps")
tf.flags.DEFINE_float("noise_precstep", 1.001, "Step of noise precision s*=ds")

tf.flags.DEFINE_integer("split_N", 30, "Increase split ratio every N steps")
tf.flags.DEFINE_float("split_ratio", 0., "Initial split ratio for conditioning")
tf.flags.DEFINE_float("split_ratio_max", 512./562, "Initial split ratio for conditioning")
tf.flags.DEFINE_integer("update_freq_post", 1, "Update frequency of posterior and sampling of new policy")

tf.flags.DEFINE_integer("kl_freq", 100, "Update kl divergence comparison")
tf.flags.DEFINE_float("kl_lambda", 10., "Weight for Kl divergence in loss")

tf.flags.DEFINE_integer("N_episodes", 4000, "Number of episodes")
tf.flags.DEFINE_integer("N_tasks", 2, "Number of tasks")
tf.flags.DEFINE_integer("L_episode", 562, "Length of episodes")
tf.flags.DEFINE_integer("num_datasets", 64, "Length of episodes")

tf.flags.DEFINE_float("tau", 1., "Update speed of target network")
tf.flags.DEFINE_integer("update_freq_target", 20, "Update frequency of target network")

tf.flags.DEFINE_integer("replay_memory_size", 1000, "Size of replay memory")
tf.flags.DEFINE_integer("iter_amax", 1, "Number of iterations performed to determine amax")
tf.flags.DEFINE_integer("save_frequency", 400, "Store images every N-th episode")
tf.flags.DEFINE_float("regularizer", 0.01, "Regularization parameter")
tf.flags.DEFINE_string('non_linearity', 'relu', 'Non-linearity used in encoder')

tf.flags.DEFINE_bool("load_model", False, "Load trained model")
tf.flags.DEFINE_integer("random_seed", 1234, "Random seed for numpy and tensorflow")

model_dir = './model/XX'

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)

np.random.seed(FLAGS.random_seed)
tf.set_random_seed(FLAGS.random_seed)

from wheel_bandit_environment import wheel_bandit_environment
from QNetwork import QNetwork
from plot_Value_fcn import plot_Value_fcn

sys.path.insert(0, './..')
from replay_buffer import replay_buffer

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
V_dir = 'figures/'+ time.strftime('%H-%M-%d_%m-%y')+ '/Valuefcn/'
if not os.path.exists(V_dir):
    os.makedirs(V_dir)

reward_dir = 'figures/'+ time.strftime('%H-%M-%d_%m-%y')+ '/'
if not os.path.exists(reward_dir):
    os.makedirs(reward_dir)

# initialize replay memory and model
fullbuffer = replay_buffer(FLAGS.replay_memory_size) # large buffer to store all experience
tempbuffer = replay_buffer(FLAGS.L_episode) # buffer for episode
rewardbuffer = replay_buffer(FLAGS.L_episode) # buffer for episode
log.info('Build Tensorflow Graph')

# ======= Dataset Generation ===========
from synthetic_data_sampler import sample_wheel_bandit_data
num_contexts = FLAGS.L_episode
num_datasets = FLAGS.num_datasets

num_actions = 5
context_dim = 2
mean_v = [1.2, 1.0, 1.0, 1.0, 1.0]
std_v = [0.01, 0.01, 0.01, 0.01, 0.01]
mu_large = 50
std_large = 0.01

# training
deltas_train = np.random.rand(num_datasets)

dataset_train = np.empty([num_datasets, num_contexts, 7])
opt_wheel_train = np.empty([num_datasets, num_contexts, 2])

for ds in range(num_datasets):
    dataset_train[ds], opt_wheel_train[ds] = sample_wheel_bandit_data(num_contexts, deltas_train[ds],
                                                                      mean_v, std_v, mu_large, std_large)

# evaluation
deltas_eval = np.array([0.5, 0.7, 0.9, 0.95, 0.99])

dataset_eval = np.empty([len(deltas_eval), num_contexts, 7])
opt_wheel_eval = np.empty([len(deltas_eval), num_contexts, 2])

for ds in range(len(deltas_eval)):
    dataset_eval[ds], opt_wheel_eval[ds] = sample_wheel_bandit_data(num_contexts, deltas_eval[ds],
                                                                      mean_v, std_v, mu_large, std_large)

# ========================================

# initialize environment
env = wheel_bandit_environment(FLAGS.action_space, FLAGS.random_seed)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    QNet = QNetwork(FLAGS, scope='QNetwork')  # neural network

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

        # save and restore model

        if episode % FLAGS.save_frequency == 0:
            # Save a checkpoint
            log.info('Save model snapshot')
            filename = os.path.join(saver_dir, 'model')
            # saver.save(sess, filename, global_step=episode, write_meta_graph=False)
            saver.save(sess, filename, global_step=episode)

        if FLAGS.load_model == True:
            saver.restore(sess, tf.train.latest_checkpoint(saver_dir))
            print('Successully restored model from '+ str(tf.train.latest_checkpoint(saver_dir)))

        # ================================================================

        # count reward
        rw = []
        r_star = 0.
        r_uni = []

        # loop tasks --------------------------------------------------------------------
        for n in range(FLAGS.N_tasks):
            # initialize buffer
            tempbuffer.reset()

            # sample mdp (theta) and shuffle context
            mdp = np.random.randint(len(deltas_train))
            shuffle_idx = np.random.permutation(np.arange(num_contexts))
            dataset_train[mdp] = dataset_train[mdp, shuffle_idx, :]
            opt_wheel_train[mdp] = opt_wheel_train[mdp, shuffle_idx, :]

            r_star += np.sum(opt_wheel_train[mdp])

            # reset posterior and sample from prior
            sess.run(QNet.reset_post)
            sess.run(QNet.sample_prior)

            # loop steps
            step = 0

            while step < FLAGS.L_episode:
                state = dataset_train[mdp, step, :2]

                # uniform reward
                r_uni.append(dataset_train[mdp,step, 2+ np.random.randint(5)])

                # take a step
                Qval = sess.run([QNet.Qout], feed_dict={QNet.state: state.reshape(-1,FLAGS.state_space)})
                action = eGreedyAction(Qval, eps)
                reward = dataset_train[mdp,step,2+ action]

                # store experience in memory
                new_experience = [state, action, reward]
                tempbuffer.add(new_experience)

                # actual reward
                rw.append(reward)

                # ----------------------------------------------------------------------
                # update (iterative)
                if (step + 1) <= np.int(split_ratio * FLAGS.L_episode):

                    # update
                    _ = sess.run([QNet.sample_post],
                             feed_dict={QNet.context_state: state.reshape(-1,2),
                                        QNet.context_action: action.reshape(-1),
                                        QNet.context_reward: reward.reshape(-1),
                                        QNet.nprec: noise_precision, QNet.is_online: True})

                # update state, and counters
                step += 1

                # -----------------------------------------------------------------------
            # append episode buffer to large buffer
            fullbuffer.add(tempbuffer.buffer)

        # append reward
        reward_episode.append(np.sum(np.array(rw)))

        # ===============================================================================
        # Gradient descent
        for e in range(batch_size):

            sess.run(QNet.reset_post)

            # sample from larger buffer [s, a, r, s', d] with current experience not yet included
            experience = fullbuffer.sample(1)

            state_sample = np.zeros((FLAGS.L_episode, FLAGS.state_space))
            action_sample = np.zeros((FLAGS.L_episode,))
            reward_sample = np.zeros((FLAGS.L_episode,))

            # fill arrays
            for k, (s0, a, r) in enumerate(experience[0]):
                state_sample[k] = s0
                action_sample[k] = a
                reward_sample[k] = r

            # split into context and prediction set
            split = np.int(split_ratio * FLAGS.L_episode * np.random.rand())

            train = np.arange(0, split)
            valid = np.arange(split, FLAGS.L_episode)

            state_train = state_sample[train, :]
            action_train = action_sample[train]
            reward_train = reward_sample[train]

            state_valid = state_sample[valid, :]
            action_valid = action_sample[valid]
            reward_valid = reward_sample[valid]

            # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            # update model
            grads, loss0 = sess.run([QNet.gradients, QNet.loss],
                feed_dict={QNet.context_state: state_train.reshape(-1,2),
                           QNet.context_action: action_train.reshape(-1),
                           QNet.context_reward: reward_train.reshape(-1),
                           QNet.state: state_valid.reshape(-1,2),
                           QNet.action: action_valid.reshape(-1),
                           QNet.reward: reward_valid.reshape(-1),
                           QNet.lr_placeholder: learning_rate,
                           QNet.nprec: noise_precision,
                           QNet.is_online: False})

            # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

            for idx, grad in enumerate(grads): # grad[0] is gradient and grad[1] the variable itself
                gradBuffer[idx] += (grad[0]/ batch_size)

            lossBuffer += loss0

        # update summary
        feed_dict = dictionary = dict(zip(QNet.gradient_holders, gradBuffer))
        feed_dict.update({QNet.lr_placeholder: learning_rate})

        # reduce summary size
        if episode % 10 == 0:
            # update summary
            _, summaries_gradvar = sess.run([QNet.updateModel, QNet.summaries_gradvar], feed_dict=feed_dict)

            loss_summary = tf.Summary(value=[tf.Summary.Value(tag='Loss', simple_value=(lossBuffer / batch_size))])
            reward_norm_summary = tf.Summary(value=[tf.Summary.Value(tag='Episodic Reward (Normalized)', simple_value=np.sum(np.array(rw)) / np.sum(np.array(r_star)))])
            regret_norm_summary = tf.Summary(value=[tf.Summary.Value(tag='Episodic Regret (normalized)',
                            simple_value=(np.sum(np.array(r_star))- np.sum(np.array(rw)))/ (np.sum(np.array(r_star)) - np.sum(np.array(r_uni))))])
            learning_rate_summary = tf.Summary(value=[tf.Summary.Value(tag='Learning rate', simple_value=learning_rate)])

            summary_writer.add_summary(loss_summary, episode)
            summary_writer.add_summary(reward_norm_summary, episode)
            summary_writer.add_summary(regret_norm_summary, episode)
            summary_writer.add_summary(summaries_gradvar, episode)
            summary_writer.add_summary(learning_rate_summary, episode)

            summary_writer.flush()
        else:
            _ = sess.run([QNet.updateModel], feed_dict=feed_dict)


        # reset buffers
        for idx in range(len(gradBuffer)):
            gradBuffer[idx] *= 0

        lossBuffer *= 0.

        # increase the batch size after the first episode. Would allow N_tasks < batch_size due to buffer
        if episode < 3:
            batch_size *= 2

        # learning rate schedule
        if learning_rate > 5e-5:
            learning_rate /= FLAGS.lr_drop

        if noise_precision < FLAGS.noise_precmax and episode % FLAGS.noise_Ndrop == 0:
            noise_precision *= FLAGS.noise_precstep

        if episode % FLAGS.split_N == 0 and episode > 0:
            split_ratio = np.min([split_ratio + 0.01, FLAGS.split_ratio_max])

        # ===============================================================
        # print to console
        if episode % FLAGS.save_frequency == 0:
            log.info('Episode %3.d with R %3.d', episode, np.sum(rw))

            print('Reward in Episode ' + str(episode) + ':   ' + str(np.sum(rw)))
            print('Learning_rate: ' + str(np.round(learning_rate, 4)) + ', Nprec: ' + str(
                np.round(noise_precision, 4)) + ', Split ratio: ' + str(np.round(split_ratio, 2)))

            # evaluation of regret to compare against bayesian bandit showdown paper ===================================
            n_shuffle = 10

            # loop over exploration parameter
            for de, deval in enumerate(deltas_eval):

                cum_regr_QN = np.zeros([n_shuffle])
                cum_regr_UF = np.zeros([n_shuffle])

                cum_rew_QN = np.zeros([n_shuffle])
                cum_rew_UF = np.zeros([n_shuffle])
                cum_rew_ST = np.zeros([n_shuffle])

                # ----------------------------------------------------------------------------------------------

                dir = V_dir + '/' + str(np.int(100 * deval)) + '/'
                if not os.path.exists(dir):
                    os.makedirs(dir)
                path = dir + 'Epoch_' + str(episode)
                plot_Value_fcn(path, deval, sess, QNet, noise_precision)

                # -----------------------------------------------------------------------------------------------

                for sh in range(n_shuffle):

                    shuffle_idx = np.random.permutation(np.arange(num_contexts))
                    dataset_eval[de] = dataset_eval[de, shuffle_idx, :]
                    opt_wheel_eval[de] = opt_wheel_eval[de, shuffle_idx, :]

                    # store expected rewards
                    rw = []
                    r_star = 0
                    r_uni = []

                    actions = []

                    # loop steps
                    step = 0

                    # sample w from prior
                    sess.run(QNet.reset_post)
                    sess.run([QNet.sample_prior])

                    r_star += np.sum(opt_wheel_eval[de, :, 0])

                    while step < num_contexts:
                        state = dataset_eval[de, step, :2]

                        # uniform reward
                        r_uni.append(dataset_eval[de,step, 2+ np.random.randint(5)])

                        # take a step
                        Qval = sess.run([QNet.Qout], feed_dict={QNet.state: state.reshape(-1, FLAGS.state_space)})
                        action = eGreedyAction(Qval, eps)

                        actions.append(action)

                        reward = dataset_eval[de,step, 2+ action]

                        # expected model reward
                        rw.append(reward)

                        # update posterior
                        if  (step + 1) <= np.int(split_ratio * FLAGS.L_episode):
                            # update
                            _ = sess.run(QNet.sample_post,
                                    feed_dict={QNet.context_state: state.reshape(-1,2),
                                               QNet.context_action: action.reshape(-1),
                                               QNet.context_reward: reward.reshape(-1),
                                               QNet.nprec: noise_precision, QNet.is_online: True})

                        # update
                        step += 1

                    cum_rew_QN[sh] = np.sum(np.array(rw))
                    cum_rew_UF[sh] = np.sum(np.array(r_uni))
                    cum_rew_ST[sh] = np.sum(np.array(r_star))

                    cum_regr_QN[sh] = np.sum(np.array(r_star)) - np.sum(np.array(rw))
                    cum_regr_UF[sh] = np.sum(np.array(r_star)) - np.sum(np.array(r_uni))

                # plot with trajectory
                path = dir + 'Epoch_' + str(episode)+ '_observations'
                buff = [dataset_eval[de, :,:2], np.asarray(actions), np.asarray(rw)]
                plot_Value_fcn(path, deval, sess, QNet, noise_precision, buff)

                # summaries
                val_rew_summary = tf.Summary(value=[tf.Summary.Value(tag='Validation Reward normalized (delta '+ str(deval)+ ')',
                                                                         simple_value=np.mean(cum_rew_QN)/ np.mean(cum_rew_ST))])
                summary_writer.add_summary(val_rew_summary, episode)

                regret_valid = cum_regr_QN / cum_regr_UF* 100
                regret_valid = np.mean(regret_valid)

                tag_name = 'Validation Cumulative Regret (delta '+ str(deval)+ ')'
                regret_norm_summary = tf.Summary(value=[tf.Summary.Value(tag=tag_name, simple_value=regret_valid)])
                summary_writer.add_summary(regret_norm_summary, episode)

            summary_writer.flush()


    # write reward to file
    df = pd.DataFrame(reward_episode)
    df.to_csv(reward_dir+'reward_per_episode', index=False)

    # reset buffers
    fullbuffer.reset()
    tempbuffer.reset()
