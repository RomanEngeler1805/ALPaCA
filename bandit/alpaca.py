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

import sys
sys.path.insert(0, './..')
from replay_buffer import replay_buffer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.16)

# General Hyperparameters
# general
tf.flags.DEFINE_integer("batch_size", 2, "Batch size for training")
tf.flags.DEFINE_float("gamma", 0., "Discount factor")
tf.flags.DEFINE_integer("N_episodes", 15001, "Number of episodes")
tf.flags.DEFINE_integer("N_tasks", 2, "Number of tasks")
tf.flags.DEFINE_integer("L_episode", 40, "Length of episodes")

# architecture
tf.flags.DEFINE_integer("hidden_space", 64, "Dimensionality of hidden space")
tf.flags.DEFINE_string('non_linearity', 'leaky_relu', 'Non-linearity used in encoder')

# domain
tf.flags.DEFINE_integer("action_space", 3, "Dimensionality of action space")
tf.flags.DEFINE_integer("state_space", 1, "Dimensionality of state space")

# posterior
tf.flags.DEFINE_float("prior_precision", 0.5, "Prior precision (1/var)")
tf.flags.DEFINE_float("noise_precision", 0.1, "Noise precision (1/var)")
tf.flags.DEFINE_float("noise_precmax", 10., "Maximum noise precision (1/var)")
tf.flags.DEFINE_integer("noise_Ndrop", 1, "Increase noise precision every N steps")
tf.flags.DEFINE_float("noise_precstep", 1.001, "Step of noise precision s*=ds")

tf.flags.DEFINE_integer("split_N", 20, "Increase split ratio every N steps")
tf.flags.DEFINE_float("split_ratio", 0., "Initial split ratio for conditioning")
tf.flags.DEFINE_float("split_ratio_max", 0., "Maximum split ratio for conditioning")
tf.flags.DEFINE_integer("update_freq_post", 1, "Update frequency of posterior and sampling of new policy")

# exploration
tf.flags.DEFINE_float("eps_initial", 0.9, "Initial value for epsilon-greedy")
tf.flags.DEFINE_float("eps_final", 0.05, "Final value for epsilon-greedy")
tf.flags.DEFINE_float("eps_step", 0.9997, "Multiplicative step for epsilon-greedy")

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

# memory
tf.flags.DEFINE_integer("replay_memory_size", 3000, "Size of replay memory")
tf.flags.DEFINE_integer("iter_amax", 1, "Number of iterations performed to determine amax")
tf.flags.DEFINE_integer("save_frequency", 1000, "Store images every N-th episode")

#
tf.flags.DEFINE_integer("random_seed", 1234, "Random seed for numpy and tensorflow")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)

np.random.seed(FLAGS.random_seed)
tf.set_random_seed(FLAGS.random_seed)

from QNetwork import QNetwork
from bandit_environment import bandit_environment
from update_model import update_model
from prioritized_memory import Memory as prioritized_replay_buffer

def eGreedyAction(x, epsilon=0.9):
    ''' select next action according to epsilon-greedy algorithm '''
    if np.random.rand() >= epsilon:
        action = np.argmax(x)
    else:
        action = np.random.randint(FLAGS.action_space)
    return action

def create_dictionary(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# Main Routine ===========================================================================
# initialize
batch_size = FLAGS.batch_size
eps = FLAGS.eps_initial
split_ratio = FLAGS.split_ratio
gamma = FLAGS.gamma
learning_rate = FLAGS.learning_rate
noise_precision = FLAGS.noise_precision

# get TF logger --------------------------------------------------------------------------
log = logging.getLogger('Train')
log.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create file handler which logs even debug messages
logger_dir = './logger/'
create_dictionary(logger_dir)

fh = logging.FileHandler(logger_dir+'tensorflow_'+ time.strftime('%H-%M-%d_%m-%y')+ '.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)

# write hyperparameters to logger
log.info('Parameters')
for key in FLAGS.__flags.keys():
    log.info('{}={}'.format(key, getattr(FLAGS, key)))

# folder to save and restore model -------------------------------------------------------
saver_dir = './model/' + time.strftime('%H-%M-%d_%m-%y') + '/'
create_dictionary(saver_dir)

# folder for plotting --------------------------------------------------------------------
base_dir = './figures/' + time.strftime('%H-%M-%d_%m-%y')
rt_dir = base_dir+ '/rt/'
create_dictionary(rt_dir)
r0_dir = base_dir+ '/r0/'
create_dictionary(r0_dir)
basis_fcn_dir = base_dir+ '/basis_fcn/'
create_dictionary(basis_fcn_dir)
reward_dir = base_dir+ '/'
create_dictionary(reward_dir)

# initialize replay memory and model
fullbuffer = prioritized_replay_buffer(FLAGS.replay_memory_size)  # large buffer to store all experience
tempbuffer = replay_buffer(FLAGS.L_episode)  # buffer for episode
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

    # report mean reward per episode
    reward_episode = []

    # timing
    time_env = []
    time_sgd = []

    # -----------------------------------------------------------------------------------
    # fill replay memory with random transitions

    for ep in range(1500):
        # episode buffer
        tempbuffer.reset()

        #environment
        env._sample_env()
        state = env._sample_state()

        step = 0
        done = False

        while (step < FLAGS.L_episode) and (done == False):
            #interact
            action = np.random.randint(FLAGS.action_space)
            next_state, reward, done, _, _ = env._step(action)

            # store experience in memory
            new_experience = [state, action, reward, next_state, done]

            # store experience in memory
            tempbuffer.add(new_experience)
            fullbuffer.add(1e3, new_experience)

            # update state
            state = next_state.copy()
            step += 1

        #fullbuffer.add(tempbuffer.buffer)

    print('Replay Buffer Filled!')

    # -----------------------------------------------------------------------------------
    # loop episodes
    print("Episodes...")
    log.info('Loop over episodes')
    for episode in range(FLAGS.N_episodes):

        # count reward
        reward_agent = []
        reward_opt = []
        reward_rand = []

        action_task = []
        entropy_episode = []

        start = time.time()

        # loop tasks --------------------------------------------------------------------
        for n in range(FLAGS.N_tasks):
            # initialize buffer
            tempbuffer.reset()

            # sample theta (i.e. bandit)
            env._sample_env()
            # resample state
            state = env._sample_state()

            # loop steps
            step = 0
            done = False

            while (step < FLAGS.L_episode) and (done == False):

                # take a step
                Qval = sess.run(QNet.Qout, feed_dict={QNet.state: state.reshape(-1,FLAGS.state_space)})[0]
                action = eGreedyAction(Qval, eps)
                next_state, reward, done, rew_max, rew_rand = env._step(action)

                Qnew = sess.run(QNet.Qout, feed_dict={QNet.state: next_state.reshape(-1, FLAGS.state_space)})[0]

                # store experience in memory
                new_experience = [state, action, reward, next_state, done]
                tempbuffer.add(new_experience)
                fullbuffer.add(np.abs(Qval[action]- reward- gamma * np.max(Qnew)), new_experience)

                # update state, and counters
                state = next_state.copy()
                step += 1

                #
                reward_agent.append(reward)
                reward_opt.append(rew_max)
                reward_rand.append(rew_rand)

                action_task.append(action)

                # -----------------------------------------------------------------------

        time_env.append(time.time() - start)

        if episode % 10 == 0:
            # tensorflow summaries
            reward_summary = tf.Summary(
                value=[tf.Summary.Value(tag='Performance/Episodic Reward',
                                        simple_value=np.sum(np.array(reward_agent)) / FLAGS.N_tasks)])
            learning_rate_summary = tf.Summary(
                value=[tf.Summary.Value(tag='Parameters/Learning rate', simple_value=learning_rate)])
            split_ratio_summary = tf.Summary(
                value=[tf.Summary.Value(tag='Parameters/Split ratio', simple_value=split_ratio)])
            epsilon_summary = tf.Summary(
                value=[tf.Summary.Value(tag='Parameters/Epsilon ratio', simple_value=eps)])
            act_entropy_summary = tf.Summary(
                value=[tf.Summary.Value(tag='Exploration-Exploitation/Entropy action',
                                        simple_value=np.mean(np.asarray(entropy_episode)))])
            regret = (np.sum(np.asarray(reward_opt)) - np.sum(np.asarray(reward_agent))) / \
                 (np.sum(np.asarray(reward_opt)) - np.sum(np.asarray(reward_rand)))
            regret_summary = tf.Summary(
                value=[tf.Summary.Value(tag='Performance/Episodic Regret', simple_value=regret)])
            summary_writer.add_summary(reward_summary, episode)
            summary_writer.add_summary(split_ratio_summary, episode)
            summary_writer.add_summary(epsilon_summary, episode)
            summary_writer.add_summary(learning_rate_summary, episode)
            summary_writer.add_summary(act_entropy_summary, episode)
            summary_writer.add_summary(regret_summary, episode)
            summary_writer.flush()

        # append reward
        reward_episode.append(np.sum(np.array(reward_agent)))

        # ==================================================================================
        start = time.time()

        for n_grad_steps in range(4):
            update_model(sess,
                         QNet,
                         Qtarget,
                         fullbuffer,
                         summary_writer,
                         FLAGS,
                         episode,
                         batch_size=batch_size* 30,
                         split_ratio=split_ratio,
                         learning_rate=learning_rate,
                         noise_precision=noise_precision)

        time_sgd.append(time.time() - start)

        # increase the batch size after the first episode. Would allow N_tasks < batch_size due to buffer
        if episode < 0:
            batch_size *= 2

        # learning rate schedule
        if learning_rate > 5e-6:
            learning_rate /= FLAGS.lr_drop

        if noise_precision < FLAGS.noise_precmax and episode % FLAGS.noise_Ndrop == 0:
            noise_precision *= FLAGS.noise_precstep

        if episode % FLAGS.split_N == 0 and episode > 0:
            split_ratio = np.min([split_ratio+ 0.01, FLAGS.split_ratio_max])

        eps = np.max([FLAGS.eps_final, eps * FLAGS.eps_step])

        # ===============================================================
        # update target network
        if episode % FLAGS.update_freq_target == 0:
            vars_modelQ = sess.run(QNet.tvars)
            feed_dict = dictionary = dict(zip(Qtarget.variable_holders, vars_modelQ))
            feed_dict.update({Qtarget.tau: FLAGS.tau})
            sess.run(Qtarget.copyParams, feed_dict=feed_dict)

        # ===============================================================
        # save model
        if episode % FLAGS.save_frequency == 0:
            # Save a checkpoint
            log.info('Save model snapshot')
            filename = os.path.join(saver_dir, 'model')
            # saver.save(sess, filename, global_step=episode, write_meta_graph=False)
            saver.save(sess, filename, global_step=episode)

        # ================================================================
        # print to console
        if episode % FLAGS.save_frequency == 0:
            log.info('Episode %3.d with R %3.d', episode, np.sum(reward_agent))

            print('AVG time env: ' + str(np.mean(np.asarray(time_env))))
            print('AVG time sgd: ' + str(np.mean(np.asarray(time_sgd))))

            print('Reward in Episode ' + str(episode) + ':   ' + str(np.sum(reward_agent)))
            print('Learning_rate: ' + str(np.round(learning_rate, 5)) + ', Nprec: ' + str(noise_precision))

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

                    # iterate
                    state = next_state.copy()
                    step += 1

                    # rewards
                    reward_agent += rew
                    reward_max += rew_max
                    reward_rand += rew_rand

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
            file.write('Episode '+ str(episode)+ '================== \n')
            file.write('Cumulative Regret\n')
            file.write( '{:3.4f}% +- {:2.4f}%\n'.format(np.mean(np.asarray(cumulative_regret)), np.std(np.asarray(cumulative_regret))))
            file.write('Simple Regret\n')
            file.write('{:3.4f}% +- {:2.4f}%\n'.format(np.mean(np.asarray(simple_regret)), np.std(np.asarray(simple_regret))))
            file.close()

    # write reward to file
    df = pd.DataFrame(reward_episode)
    df.to_csv(reward_dir+'reward_per_episode', index=False)

    # reset buffers
    fullbuffer.reset()
    tempbuffer.reset()
