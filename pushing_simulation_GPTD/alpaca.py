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
import sys
from matplotlib import ticker
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.13)

# General Hyperparameters
# general
tf.flags.DEFINE_integer("batch_size", 2, "Batch size for training")
tf.flags.DEFINE_float("gamma", 0.95, "Discount factor")
tf.flags.DEFINE_integer("N_episodes", 15000, "Number of episodes")
tf.flags.DEFINE_integer("N_tasks", 2, "Number of tasks")
tf.flags.DEFINE_integer("L_episode", 25, "Length of episodes")

# architecture
tf.flags.DEFINE_integer("hidden_space", 64, "Dimensionality of hidden space")
tf.flags.DEFINE_integer("latent_space", 4, "Dimensionality of latent space")
tf.flags.DEFINE_string('non_linearity', 'leaky_relu', 'Non-linearity used in encoder')
tf.flags.DEFINE_integer("nstep", 1, "n-step TD return")

# domain
tf.flags.DEFINE_integer("action_space", 5, "Dimensionality of action space")  # only x-y currently
tf.flags.DEFINE_integer("state_space", 6, "Dimensionality of state space")  # [x,y,theta,vx,vy,vtheta]

# posterior
tf.flags.DEFINE_float("prior_precision", 0.1, "Prior precision (1/var)")
tf.flags.DEFINE_float("noise_precision", 0.01, "Noise precision (1/var)")
tf.flags.DEFINE_float("noise_precmax", 0.1, "Maximum noise precision (1/var)")
tf.flags.DEFINE_integer("noise_Ndrop", 1, "Increase noise precision every N steps")
tf.flags.DEFINE_float("noise_precstep", 1.005, "Step of noise precision s*=ds")

tf.flags.DEFINE_integer("split_N", 20, "Increase split ratio every N steps")
tf.flags.DEFINE_float("split_ratio", 0., "Initial split ratio for conditioning")
tf.flags.DEFINE_float("split_ratio_max", 0.0, "Maximum split ratio for conditioning")
tf.flags.DEFINE_integer("update_freq_post", 10, "Update frequency of posterior and sampling of new policy")

# exploration
tf.flags.DEFINE_float("eps_initial", 0.0, "Initial value for epsilon-greedy")
tf.flags.DEFINE_float("eps_final", 0.0, "Final value for epsilon-greedy")
tf.flags.DEFINE_float("eps_step", 0.9997, "Multiplicative step for epsilon-greedy")

# target
tf.flags.DEFINE_float("tau", 0.01, "Update speed of target network")
tf.flags.DEFINE_integer("update_freq_target", 1, "Update frequency of target network")

# loss
tf.flags.DEFINE_float("learning_rate", 2e-3, "Initial learning rate") # X
tf.flags.DEFINE_float("lr_drop", 1.0003, "Drop of learning rate per episode")
tf.flags.DEFINE_float("grad_clip", 1e4, "Absolute value to clip gradients")
tf.flags.DEFINE_float("huber_d", 1e1, "Switch point from quadratic to linear")
tf.flags.DEFINE_float("regularizer", 1e-2, "Regularization parameter") # X

# reward
tf.flags.DEFINE_float("rew_norm", 1e0, "Normalization factor for reward")

# memory
tf.flags.DEFINE_integer("replay_memory_size", 30000, "Size of replay memory")
tf.flags.DEFINE_integer("iter_amax", 1, "Number of iterations performed to determine amax")
tf.flags.DEFINE_integer("save_frequency", 200, "Store images every N-th episode")

#
tf.flags.DEFINE_integer("random_seed", 2345, "Random seed for numpy and tensorflow")
tf.flags.DEFINE_bool("load_model", False, "Load trained model")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)

np.random.seed(FLAGS.random_seed)
tf.set_random_seed(FLAGS.random_seed)

from push_env import PushEnv
from QNetwork import QNetwork
from generate_plots import generate_plots, value_function_plot, policy_plot#, generate_posterior_plots, policy_plot
from update_model import update_model
#
from replay_buffer import replay_buffer
from prioritized_memory import Memory as prioritized_replay_buffer

def eGreedyAction(x, epsilon=0.):
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
#
batch_size = FLAGS.batch_size
eps = FLAGS.eps_initial
split_ratio = FLAGS.split_ratio
nstep = FLAGS.nstep
gamma = FLAGS.gamma

# get TF logger --------------------------------------------------------------------------
log = logging.getLogger('Train')
log.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create file handler which logs even debug messages
logger_dir = './logger/'
create_dictionary(logger_dir)

fh = logging.FileHandler(logger_dir + 'tensorflow_' + time.strftime('%H-%M-%d_%m-%y') + '.log')
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
loader_dir = './model/XX/'

# folder for plotting --------------------------------------------------------------------
base_dir = './figures/' + time.strftime('%H-%M-%d_%m-%y')
reward_dir = base_dir + '/Reward/'
create_dictionary(reward_dir)
trajectory_dir = base_dir + '/Trajectories/'
create_dictionary(trajectory_dir)
Qprior = base_dir + '/Qprior/'
create_dictionary(Qprior)
Qposterior_dir = base_dir + '/Qposterior/'
create_dictionary(Qposterior_dir)
policy_dir = base_dir + '/Policy/'
create_dictionary(policy_dir)

# initialize replay memory and model
fullbuffer = prioritized_replay_buffer(FLAGS.replay_memory_size)  # large buffer to store all experience
tempbuffer = replay_buffer(FLAGS.L_episode)  # buffer for episode
log.info('Build Tensorflow Graph')

# initialize environment
env = PushEnv()

# initialize
learning_rate = FLAGS.learning_rate
noise_precision = FLAGS.noise_precision

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

    summary_writer = tf.summary.FileWriter(logdir=os.path.join('./', 'summaries/', time.strftime('%H-%M-%d_%m-%y')),
                                           graph=sess.graph)

    #
    if FLAGS.load_model == True:
        loader = tf.train.Saver(max_to_keep=4)
        loader.restore(sess, tf.train.latest_checkpoint(loader_dir))
        print('Successully restored model from ' + str(tf.train.latest_checkpoint(loader_dir)))

    # report mean reward per episode
    reward_episode = []

    # timing
    time_env = []
    time_sgd = []

    # -----------------------------------------------------------------------------------
    # fill replay memory with random transitions

    for ep in range(500):
        # episode buffer
        tempbuffer.reset()

        #environment
        state = env.reset()

        # network
        sess.run(QNet.sample_prior)

        step = 0
        done = False

        while (step < FLAGS.L_episode) and (done == False):
            #interact
            action = np.random.randint(5)
            next_state, reward, done, _ = env.step(action)

            # store experience in memory
            new_experience = [state, action, reward, next_state, nstep, done]

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
        rw = []
        action_task = []
        entropy_episode = []
        target_distance = []
        max_speed = []

        start = time.time()

        # loop tasks --------------------------------------------------------------------
        for n in range(FLAGS.N_tasks):
            # initialize buffer
            tempbuffer.reset()

            # reset environment
            state = env.reset()

            # network
            sess.run(QNet.sample_prior)

            # loop steps
            step = 0
            done = False

            # n-step reward
            nreward = []
            nstate = []
            naction = []
            nQold = []
            discount_factor = np.asarray([gamma**t for t in range(0, nstep)])

            while (step < FLAGS.L_episode) and (done == False):

                # take a step
                Qval = sess.run(QNet.Qout, feed_dict={QNet.state: state.reshape(-1,FLAGS.state_space)})[0]
                action = eGreedyAction(Qval, eps)

                next_state, reward, done, _ = env.step(action)

                nstate.append(state)
                nreward.append(reward)
                naction.append(action)
                nQold.append(Qval[action])

                if step >= nstep-1:
                    # store experience in memory
                    # reward has different indices since for the prioritized replay a 0 is added
                    discounted_reward = np.dot(discount_factor, np.asarray(nreward[step-nstep+1:]))

                    exponent = nstep

                    new_experience = [nstate[step-nstep+1],
                                      naction[step-nstep+1],
                                      discounted_reward,
                                      next_state,
                                      exponent,
                                      done]

                    Qnew = sess.run(QNet.Qout, feed_dict={QNet.state: next_state.reshape(-1,FLAGS.state_space)})[0]
                    action_new = eGreedyAction(Qnew, eps)

                    # store experience in memory
                    tempbuffer.add(new_experience)
                    fullbuffer.add(np.abs(nQold[step-nstep+1]- gamma**exponent* Qnew[action_new]- discounted_reward), new_experience)

                # actual reward
                rw.append(reward* FLAGS.rew_norm)
                action_task.append(action)

                # update state, and counters
                state = next_state.copy()
                step += 1

                '''
                if episode % FLAGS.save_frequency == 0 and n == 0 and step == 0:
                    generate_posterior_plots(sess, QNet, base_dir, episode, FLAGS.param_case, step)
                    policy_plot(sess, QNet, tempbuffer, FLAGS, episode, FLAGS.param_case, step, base_dir)
                '''
                # -----------------------------------------------------------------------

            upper_limit = np.min([step+ nstep- 1, FLAGS.L_episode+ nstep- 1])
            for sstep in range(step, upper_limit):
                discount_factor = np.asarray([gamma ** t for t in range(0, upper_limit- sstep)])
                discounted_reward = np.dot(discount_factor, np.asarray(nreward[sstep - nstep + 1:]))

                exponent = upper_limit - sstep

                new_experience = [nstate[sstep - nstep + 1],
                                  naction[sstep - nstep + 1],
                                  discounted_reward,
                                  next_state,
                                  exponent,
                                  done]

                # store experience in memory
                tempbuffer.add(new_experience)
                fullbuffer.add(np.abs(nQold[sstep - nstep + 1] - gamma ** exponent * Qnew[action_new] - discounted_reward), new_experience)

            # append episode buffer to large buffer
            #fullbuffer.add(tempbuffer.buffer)

            # entropy of action selection
            _, action_count = np.unique(np.asarray(action_task), return_counts=True)
            action_prob = 1.*action_count / np.sum(action_count)
            entropy_episode.append(np.sum([-p*np.log(p) for p in action_prob if p != 0.]))

            # final distance to target
            target_distance.append(np.linalg.norm(env.target_position- state[4:6]))

            # maximum speed of robot arm
            nstate = np.asarray(nstate)

            speed = np.linalg.norm(nstate[1:, :2] - nstate[:-1, :2], axis=1)
            max_speed.append(np.max(speed) * env.control_hz)

        time_env.append(time.time() - start)

        if episode % 10 == 0:
            # tensorflow summaries
            reward_summary = tf.Summary(
                value=[tf.Summary.Value(tag='Performance2/Episodic Reward',
                                        simple_value=np.sum(np.array(rw)) / FLAGS.N_tasks)])
            distance_summary = tf.Summary(
                value=[tf.Summary.Value(tag='Performance2/Target Distance',
                                        simple_value=np.mean(np.asarray(target_distance)) / FLAGS.N_tasks)])
            speed_summary = tf.Summary(
                value=[tf.Summary.Value(tag='Performance2/Max Speed',
                                        simple_value=np.mean(np.asarray(max_speed)) / FLAGS.N_tasks)])
            learning_rate_summary = tf.Summary(
                value=[tf.Summary.Value(tag='Parameters/Learning rate', simple_value=learning_rate)])
            split_ratio_summary = tf.Summary(
                value=[tf.Summary.Value(tag='Parameters/Split ratio', simple_value=split_ratio)])
            epsilon_summary = tf.Summary(
                value=[tf.Summary.Value(tag='Parameters/Epsilon ratio', simple_value=eps)])
            act_entropy_summary = tf.Summary(
                value=[tf.Summary.Value(tag='Exploration-Exploitation/Entropy action', simple_value=np.mean(np.asarray(entropy_episode)))])
            len_traj_summary = tf.Summary(
                value=[tf.Summary.Value(tag='Exploration-Exploitation/Trajectory Length', simple_value=step)])
            summary_writer.add_summary(reward_summary, episode)
            summary_writer.add_summary(distance_summary, episode)
            summary_writer.add_summary(speed_summary, episode)
            summary_writer.add_summary(split_ratio_summary, episode)
            summary_writer.add_summary(epsilon_summary, episode)
            summary_writer.add_summary(learning_rate_summary, episode)
            summary_writer.add_summary(act_entropy_summary, episode)
            summary_writer.add_summary(len_traj_summary, episode)
            summary_writer.flush()

        # reward in episode
        reward_episode.append(np.sum(np.array(rw)) / FLAGS.N_tasks)

        # visual inspection ================================================================
        if episode % FLAGS.save_frequency == 0:
            generate_plots(sess, summary_writer, base_dir, tempbuffer, FLAGS, episode)

        eps = np.max([FLAGS.eps_final, eps*FLAGS.eps_step])

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
                         batch_size=batch_size*30,
                         split_ratio=split_ratio,
                         learning_rate=learning_rate,
                         noise_precision=noise_precision)

        time_sgd.append(time.time()- start)

        # increase the batch size after the first episode. Would allow N_tasks < batch_size due to buffer
        if episode < 0:
            batch_size *= 2

        # learning rate schedule
        if learning_rate > 5e-5:
            learning_rate /= FLAGS.lr_drop

        if noise_precision < FLAGS.noise_precmax and episode % FLAGS.noise_Ndrop == 0:
            noise_precision *= FLAGS.noise_precstep

        if episode % FLAGS.split_N == 0 and episode > 500:
            split_ratio = np.min([split_ratio + 0.003, FLAGS.split_ratio_max])

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
            log.info('Episode %3.d with R %3.d', episode, np.sum(rw))

            print('AVG time env: '+ str(np.mean(np.asarray(time_env))))
            print('AVG time sgd: ' + str(np.mean(np.asarray(time_sgd))))

            print('Reward in Episode ' + str(episode) + ':   ' + str(np.sum(rw)))
            print('Learning_rate: ' + str(np.round(learning_rate, 5)) + ', Nprec: ' + str(
                noise_precision) + ', Split ratio: ' + str(np.round(split_ratio, 2)))

            #evaluation_plots(sess, QNet, env, tempbuffer, FLAGS, summary_writer, noise_precision, episode, split_ratio, base_dir)
            value_function_plot(sess, QNet, tempbuffer, FLAGS, episode, base_dir)
            policy_plot(sess, QNet, tempbuffer, FLAGS, episode, base_dir)


    # write reward to file
    df = pd.DataFrame(reward_episode)
    df.to_csv(reward_dir + 'reward_per_episode', index=False)

    # reset buffers
    fullbuffer.reset()
    tempbuffer.reset()
