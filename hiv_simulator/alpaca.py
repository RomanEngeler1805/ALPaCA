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
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)

# General Hyperparameters
tf.flags.DEFINE_integer("batch_size", 2, "Batch size for training")
tf.flags.DEFINE_integer("action_space", 4, "Dimensionality of action space")  # only x-y currently
tf.flags.DEFINE_integer("state_space", 6, "Dimensionality of state space")  # [x,y,theta,vx,vy,vtheta]
tf.flags.DEFINE_integer("hidden_space", 128, "Dimensionality of hidden space")
tf.flags.DEFINE_integer("latent_space", 22, "Dimensionality of latent space")
tf.flags.DEFINE_float("gamma", 0.95, "Discount factor")

tf.flags.DEFINE_float("learning_rate", 5e-3, "Initial learning rate") # X
tf.flags.DEFINE_float("lr_drop", 1.001, "Drop of learning rate per episode")

tf.flags.DEFINE_float("prior_precision", 0.1, "Prior precision (1/var)")
tf.flags.DEFINE_float("noise_precision", 0.01, "Noise precision (1/var)")
tf.flags.DEFINE_float("noise_precmax", 100, "Maximum noise precision (1/var)")
tf.flags.DEFINE_integer("noise_Ndrop", 1, "Increase noise precision every N steps")
tf.flags.DEFINE_float("noise_precstep", 1.0001, "Step of noise precision s*=ds")

tf.flags.DEFINE_integer("split_N", 20, "Increase split ratio every N steps")
tf.flags.DEFINE_float("split_ratio", 0., "Initial split ratio for conditioning")
tf.flags.DEFINE_float("split_ratio_max", 0.0, "Initial split ratio for conditioning")
tf.flags.DEFINE_integer("update_freq_post", 10, "Update frequency of posterior and sampling of new policy")

tf.flags.DEFINE_integer("N_episodes", 4000, "Number of episodes")
tf.flags.DEFINE_integer("N_tasks", 2, "Number of tasks")
tf.flags.DEFINE_integer("L_episode", 200, "Length of episodes")

tf.flags.DEFINE_float("tau", 0.01, "Update speed of target network")
tf.flags.DEFINE_integer("update_freq_target", 1, "Update frequency of target network")

tf.flags.DEFINE_float("rew_norm", 1e4, "Normalization factor for reward") # X

tf.flags.DEFINE_integer("replay_memory_size", 1000, "Size of replay memory")
tf.flags.DEFINE_integer("iter_amax", 1, "Number of iterations performed to determine amax")
tf.flags.DEFINE_integer("save_frequency", 200, "Store images every N-th episode")
tf.flags.DEFINE_float("regularizer", 1e-3, "Regularization parameter") # X
tf.flags.DEFINE_string('non_linearity', 'leaky_relu', 'Non-linearity used in encoder')

tf.flags.DEFINE_integer("param_case", 5, "Which of the 6 test param sets")

tf.flags.DEFINE_integer("random_seed", 1234, "Random seed for numpy and tensorflow")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)

np.random.seed(FLAGS.random_seed)
tf.set_random_seed(FLAGS.random_seed)

from hiv import HIVTreatment
from QNetwork import QNetwork

sys.path.insert(0, './..')
from replay_buffer import replay_buffer
#from prioritized_memory import Memory


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
eps = 0.1
split_ratio = FLAGS.split_ratio

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

# folder for plotting --------------------------------------------------------------------
reward_dir = './figures/' + time.strftime('%H-%M-%d_%m-%y') + '/Reward_Histogram/'
create_dictionary(reward_dir)
trajectory_dir = './figures/' + time.strftime('%H-%M-%d_%m-%y') + '/Trajectories/'
create_dictionary(trajectory_dir)
histogram_dir = './figures/' + time.strftime('%H-%M-%d_%m-%y') + '/Action_Histogram/'
create_dictionary(histogram_dir)
states_dir = './figures/' + time.strftime('%H-%M-%d_%m-%y') + '/State_Histogram/'
create_dictionary(states_dir)
Q_Sum_r_dir = './figures/' + time.strftime('%H-%M-%d_%m-%y') + '/Q_Sum_r/'
create_dictionary(Q_Sum_r_dir)
V_E_dir = './figures/' + time.strftime('%H-%M-%d_%m-%y') + '/Marker_V_E/'
create_dictionary(V_E_dir)

# initialize replay memory and model
fullbuffer = replay_buffer(FLAGS.replay_memory_size)  # large buffer to store all experience
tempbuffer = replay_buffer(FLAGS.L_episode)  # buffer for episode
log.info('Build Tensorflow Graph')

# initialize environment
env = HIVTreatment(rew_norm= FLAGS.rew_norm)

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

    # initialize
    learning_rate = FLAGS.learning_rate
    noise_precision = FLAGS.noise_precision

    gradBuffer = sess.run(QNet.tvars) # get shapes of tensors

    for idx in range(len(gradBuffer)):
        gradBuffer[idx] *= 0

    # loss buffers to visualize in tensorboard
    lossBuffer = 0.
    loss1Buffer = 0.
    loss2Buffer = 0.
    lossregBuffer = 0.

    # report mean reward per episode
    reward_episode = []

    with open('./' + 'hiv' + '_preset_hidden_params', 'r') as f:
        preset_parameters = pickle.load(f)

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

            # reset environment
            env.reset()
            preset_hidden_params = preset_parameters[episode%6]
            env.param_set = preset_hidden_params
            state = env.observe()

            # sample w from prior TODO (automatically resets posterior)
            sess.run([QNet.reset_post])
            sess.run([QNet.sample_prior])

            # loop steps

            step = 0
            flag = True

            while step < FLAGS.L_episode:

                # take a step
                Qval = sess.run([QNet.Qout], feed_dict={QNet.state: state.reshape(-1,FLAGS.state_space)})
                action = eGreedyAction(Qval, eps)

                next_state, reward, done = env.step(action, perturb_params=True)

                # check validity of trajectory
                # TODO fix to e.g. 2* range of standard parameters
                if any(next_state< -10.) or any(next_state> 20.) or done:
                    print(next_state)
                    flag = False
                    break

                # store experience in memory
                new_experience = [state, action, reward, next_state, done]

                # store experience in memory
                tempbuffer.add(new_experience)

                # actual reward
                rw.append(reward)

                # update state, and counters
                state = next_state.copy()
                step += 1

                _ , _ = sess.run([QNet.w_assign, QNet.L_assign],
                                     feed_dict={QNet.context_state: state.reshape(-1, FLAGS.state_space),
                                                QNet.context_action: np.array(action).reshape(-1),
                                                QNet.context_reward: np.array(reward).reshape(-1),
                                                QNet.context_state_next: next_state.reshape(-1, FLAGS.state_space),
                                                QNet.context_done: np.array(done).reshape(-1, 1),
                                                QNet.nprec: noise_precision, QNet.is_online: True})

                # update posterior iteratively
                if (step + 1) % FLAGS.update_freq_post == 0 and (step + 1) <= np.int(split_ratio * FLAGS.L_episode):
                    # update
                    __ = sess.run([QNet.sample_post])
                    # -----------------------------------------------------------------------

                # -----------------------------------------------------------------------
            if flag:
                # append episode buffer to large buffer
                fullbuffer.add(tempbuffer.buffer)

        # reward in episode
        reward_episode.append(np.sum(np.array(rw)) / FLAGS.N_tasks)

        # visual inspection ================================================================
        if episode % FLAGS.save_frequency == 0:
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

            plt.figure()
            plt.plot(state_train)
            plt.xlabel('time')
            plt.ylabel('Concentration')
            plt.legend(['T1', 'T2', 'T1s', 'T2s', 'V', 'E'], loc='upper right')
            plt.title( str(np.int(np.sum(np.array(rw)) / FLAGS.N_tasks)))
            plt.savefig(trajectory_dir+ 'Episode_'+ str(episode))
            plt.close()

            plt.figure()
            plt.hist(action_train)
            plt.xlabel('action')
            plt.ylabel('count')
            plt.title(str(np.int(np.sum(np.array(rw)) / FLAGS.N_tasks)))
            plt.savefig(histogram_dir + 'Episode_' + str(episode))
            plt.close()

            plt.figure()
            plt.hist(reward_train)
            plt.xlabel('reward')
            plt.ylabel('count')
            plt.title(str(np.int(np.sum(np.array(rw)) / FLAGS.N_tasks)))
            plt.savefig(reward_dir + 'Episode_' + str(episode))
            plt.close()

            fig, ax = plt.subplots(ncols=5)
            for i in range(5):
                ax[i].hist(state_train[:,i])
                ax[i].set_xlabel('state '+str(i))
                ax[i].set_ylabel('count')
            plt.title(str(np.int(np.sum(np.array(rw)) / FLAGS.N_tasks)))
            plt.savefig(states_dir + 'Episode_' + str(episode))
            plt.close()

        if episode % 50 == 0:
            eps = np.max([0.1, eps*0.97])
            print(eps)

        # ==================================================================================

        # Gradient descent
        for e in range(batch_size):

            # probably not necessary: check
            sess.run(QNet.reset_post)

            # sample from larger buffer [s, a, r, s', d] with current experience not yet included
            experience = fullbuffer.sample(1)

            L_episode = len(experience[0])

            state_sample = np.zeros((L_episode, FLAGS.state_space))
            action_sample = np.zeros((L_episode,))
            reward_sample = np.zeros((L_episode,))
            next_state_sample = np.zeros((L_episode, FLAGS.state_space))
            done_sample = np.zeros((L_episode,))

            # fill arrays
            for k, (s0, a, r, s1, d) in enumerate(experience[0]):
                state_sample[k] = s0
                action_sample[k] = a
                reward_sample[k] = r
                next_state_sample[k] = s1
                done_sample[k] = d

            # split into context and prediction set
            split = np.int(split_ratio * L_episode * np.random.rand())

            train = np.arange(0, split)
            valid = np.arange(split, L_episode)

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

            # TODO: this part is very inefficient due to many session calls and processing data multiple times
            # select amax from online network
            amax_online = sess.run(QNet.max_action,
                                   feed_dict={QNet.context_state: state_train.reshape(-1, FLAGS.state_space),
                                              QNet.context_action: action_train.reshape(-1),
                                              QNet.context_reward: reward_train.reshape(-1),
                                              QNet.context_state_next: next_state_train.reshape(-1, FLAGS.state_space),
                                              QNet.context_done: done_train.reshape(-1,1),
                                              QNet.state: state_valid,
                                              QNet.state_next: next_state_valid,
                                              QNet.nprec: noise_precision,
                                              QNet.is_online: False})

            # evaluate target model
            phi_max_target = sess.run(Qtarget.phi_max,
                                   feed_dict={Qtarget.context_state: state_train,
                                              Qtarget.context_action: action_train,
                                              Qtarget.context_reward: reward_train,
                                              Qtarget.context_state_next: next_state_train,
                                              Qtarget.state: state_valid,
                                              Qtarget.state_next: next_state_valid,
                                              Qtarget.amax_online: amax_online})

            # update model
            grads, loss, loss1, loss2, lossreg, Qdiff = sess.run(
                [QNet.gradients, QNet.loss, QNet.loss1, QNet.loss2, QNet.loss_reg, QNet.Qdiff],
                feed_dict={QNet.context_state: state_train.reshape(-1, FLAGS.state_space),
                           QNet.context_action: action_train.reshape(-1),
                           QNet.context_reward: reward_train.reshape(-1),
                           QNet.context_state_next: next_state_train.reshape(-1, FLAGS.state_space),
                           QNet.context_done: done_train.reshape(-1,1),
                           QNet.state: state_valid,
                           QNet.action: action_valid,
                           QNet.reward: reward_valid,
                           QNet.state_next: next_state_valid,
                           QNet.done: done_valid,
                           QNet.phi_max_target: phi_max_target,
                           QNet.amax_online: amax_online,
                           QNet.lr_placeholder: learning_rate,
                           QNet.nprec: noise_precision,
                           QNet.is_online: False})


            # fullbuffer.update(idxs[0], loss0/ len(valid))

            for idx, grad in enumerate(grads): # grad[0] is gradient and grad[1] the variable itself
                gradBuffer[idx] += (grad[0]/ batch_size)

            lossBuffer += loss
            loss1Buffer += loss1
            loss2Buffer += loss2
            lossregBuffer += lossreg


        # update summary
        feed_dict = dictionary = dict(zip(QNet.gradient_holders, gradBuffer))
        feed_dict.update({QNet.lr_placeholder: learning_rate})

        if episode > 1:
            # reduce summary size
            if episode % 10 == 0:
                # update summary
                _, summaries_gradvar = sess.run([QNet.updateModel, QNet.summaries_gradvar], feed_dict=feed_dict)

                loss_summary = tf.Summary(value=[tf.Summary.Value(tag='Loss', simple_value=(lossBuffer / batch_size))])
                loss1_summary = tf.Summary(value=[tf.Summary.Value(tag='Loss1', simple_value=(loss1Buffer / batch_size))])
                loss2_summary = tf.Summary(value=[tf.Summary.Value(tag='Loss2', simple_value=(loss2Buffer / batch_size))])
                lossreg_summary = tf.Summary(value=[tf.Summary.Value(tag='Loss reg', simple_value=(lossregBuffer / batch_size))])
                reward_summary = tf.Summary(value=[
                    tf.Summary.Value(tag='Episodic Reward', simple_value=np.sum(np.array(rw))* FLAGS.rew_norm /FLAGS.N_tasks)])
                learning_rate_summary = tf.Summary(
                    value=[tf.Summary.Value(tag='Learning rate', simple_value=learning_rate)])

                summary_writer.add_summary(loss_summary, episode)
                summary_writer.add_summary(loss1_summary, episode)
                summary_writer.add_summary(loss2_summary, episode)
                summary_writer.add_summary(lossreg_summary, episode)
                summary_writer.add_summary(reward_summary, episode)
                summary_writer.add_summary(summaries_gradvar, episode)
                summary_writer.add_summary(learning_rate_summary, episode)

                summary_writer.flush()
            else:
                _ = sess.run([QNet.updateModel], feed_dict=feed_dict)


        # reset buffers
        for idx in range(len(gradBuffer)):
            gradBuffer[idx] *= 0

        lossBuffer *= 0.
        loss1Buffer *= 0.
        loss2Buffer *= 0.
        lossregBuffer *= 0.

        # increase the batch size after the first episode. Would allow N_tasks < batch_size due to buffer
        if episode < 4:
            batch_size *= 2

        # learning rate schedule
        if learning_rate > 5e-5:
            learning_rate /= FLAGS.lr_drop

        if noise_precision < FLAGS.noise_precmax and episode % FLAGS.noise_Ndrop == 0:
            noise_precision *= FLAGS.noise_precstep

        if episode % FLAGS.split_N == 0 and episode > 0:
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

        # if episode > 0 and episode % 1000 == 0:
        # Save a checkpoint
        # log.info('Save model snapshot')

        # filename = os.path.join(saver_dir, 'model')
        # saver.save(sess, filename, global_step=episode, write_meta_graph=False)

        # ================================================================
        # print to console
        if episode % FLAGS.save_frequency == 0:
            log.info('Episode %3.d with R %3.d', episode, np.sum(rw))

            print('Reward in Episode ' + str(episode) + ':   ' + str(np.sum(rw)))
            print('Learning_rate: ' + str(np.round(learning_rate, 5)) + ', Nprec: ' + str(
                noise_precision) + ', Split ratio: ' + str(np.round(split_ratio, 2)))

            # ==================================================== #
            # log history of Q values, history of discounted rewards -> Qlog, Rlog
            # log V, E markers -> Vlog, Elog
            # log reward -> Rlog
            # ==================================================== #
            Neval = 3 # number of repeat

            # evaluation ------------------------------------------------
            with open('./'+'hiv'+'_preset_hidden_params','r') as f:
                preset_parameters = pickle.load(f)

            for i_eval in range(6):

                # logging
                Qlog = np.zeros([Neval, FLAGS.L_episode])
                Rlog = np.zeros([Neval, FLAGS.L_episode])
                Vlog = np.zeros([Neval, FLAGS.L_episode])
                Elog = np.zeros([Neval, FLAGS.L_episode])

                # load hidden parameters for evaluation
                preset_hidden_params = preset_parameters[i_eval]

                for ne in range(Neval):

                    # reset buffer
                    tempbuffer.reset()
                    # reset environment
                    env.reset()
                    env.param_set = preset_hidden_params
                    state = env.observe()

                    # sample w from prior
                    sess.run([QNet.reset_post])
                    sess.run([QNet.sample_prior])

                    # loop steps
                    step = 0

                    while step < FLAGS.L_episode:

                        # take a step
                        Qval = sess.run([QNet.Qout], feed_dict={QNet.state: state.reshape(-1, FLAGS.state_space)})
                        action = eGreedyAction(Qval, eps)
                        next_state, reward, done = env.step(action, perturb_params=True)

                        # store experience in memory
                        new_experience = [state, action, reward, next_state, done]
                        tempbuffer.add(new_experience)

                        # logging
                        Qlog[ne, step] = np.max(Qval)
                        Rlog[ne, step] = reward
                        Vlog[ne, step] = state[4]
                        Elog[ne, step] = state[5]

                        if done == 1:
                            break

                        _, _ = sess.run([QNet.w_assign, QNet.L_assign],
                                        feed_dict={QNet.context_state: state.reshape(-1, FLAGS.state_space),
                                                   QNet.context_action: np.array(action).reshape(-1),
                                                   QNet.context_reward: np.array(reward).reshape(-1),
                                                   QNet.context_state_next: next_state.reshape(-1, FLAGS.state_space),
                                                   QNet.context_done: np.array(done).reshape(-1, 1),
                                                   QNet.nprec: noise_precision, QNet.is_online: True})

                        # update posterior iteratively
                        if (step + 1) % FLAGS.update_freq_post == 0 and (step + 1) <= np.int(split_ratio * FLAGS.L_episode):
                            # update
                            __ = sess.run([QNet.sample_post])

                        # update state, and counters
                        state = next_state.copy()
                        step += 1

                    # -----------------------------------------------------------------------

                reward_tensorboard = np.mean(np.sum(Rlog, axis=1))* FLAGS.rew_norm

                # log to tensorboard
                reward_eval_summary = tf.Summary(value=[tf.Summary.Value(tag='Reward Eval '+str(i_eval), simple_value=reward_tensorboard)])
                summary_writer.add_summary(reward_eval_summary, episode)
                summary_writer.flush()

                # plot for visual inspection
                discounted_r = np.zeros_like(Rlog[0], dtype=np.float32)
                running_add = 0
                for t in reversed(range(len(Rlog[0]))):
                    running_add = running_add * FLAGS.gamma + Rlog[0,t]
                    discounted_r[t] = running_add

                plt.figure()
                plt.scatter(discounted_r, Qlog[0])
                plt.xlabel('summed discounted reward')
                plt.ylabel('Q value')
                plt.savefig(Q_Sum_r_dir+ 'Episode_'+ str(episode)+'_case_'+ str(i_eval))
                plt.close()

                plt.figure()
                plt.scatter(Vlog[0], Elog[0])
                plt.xlabel('V')
                plt.ylabel('E')
                plt.xlim([-1, 10])
                plt.ylim([-1, 10])
                plt.savefig(V_E_dir + 'Episode_' + str(episode)+'_case_'+ str(i_eval))
                plt.close()


    # write reward to file
    df = pd.DataFrame(reward_episode)
    df.to_csv(reward_dir + 'reward_per_episode', index=False)

    # reset buffers
    fullbuffer.reset()
    tempbuffer.reset()
