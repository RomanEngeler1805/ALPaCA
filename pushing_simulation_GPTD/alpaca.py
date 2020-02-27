'''
Class structure for ALPaCA meta-RL code
'''

import numpy as np
import tensorflow as tf
import os
import time
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pandas as pd
import sys
from matplotlib import ticker
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.13)


from push_env import PushEnv
from QNetwork import QNetwork
from utils.update_model import update_model # recall to optimizer?

from utils.generate_plots import generate_plots, value_function_plot, policy_plot, generate_posterior_plots, evaluate_Q#, policy_plot
#
from utils.replay_buffer import replay_buffer
from utils.prioritized_memory import Memory as prioritized_replay_buffer

def create_dictionary(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)



class alpaca:

    def __init__(self, FLAGS):
        '''
        initialize parameters
        initialize model
        fill replay buffer
        '''

        # fixed hyperparameters
        self.FLAGS = FLAGS
        # adaptable hyperparameters
        self.batch_size = FLAGS.batch_size
        self.split_ratio = FLAGS.split_ratio
        self.gamma = FLAGS.gamma
        self.learning_rate = FLAGS.learning_rate
        self.noise_precision = FLAGS.noise_precision

        # environment
        self.env = PushEnv()
        self.env.rew_scale = FLAGS.rew_norm

        # memory
        self.fullbuffer = prioritized_replay_buffer(FLAGS.replay_memory_size)  # large buffer to store all experience
        self.tempbuffer = replay_buffer(FLAGS.L_episode)  # standard buffer for training (episode)
        self.evalbuffer = replay_buffer(FLAGS.L_episode)  # standard buffer for validation

        # model
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.Qmain = QNetwork(FLAGS, scope='QNetwork')  # neural network
        self.Qtarget = QNetwork(FLAGS, scope='TargetNetwork')
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # checkpoint
        self.saver = tf.train.Saver(max_to_keep=6)
        self.summary_writer = tf.summary.FileWriter(logdir=os.path.join('./', 'summaries/', time.strftime('%H-%M-%d_%m-%y')),
                                                    graph=self.sess.graph)
        # supporting ------------------------------------------------------------------------
        # logger
        logger_dir = './logger/'
        create_dictionary(logger_dir)
        self.log = logging.getLogger('Train')
        self.log.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(logger_dir + 'tensorflow_' + time.strftime('%H-%M-%d_%m-%y') + '.log')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        self.log.addHandler(fh)

        # write hyperparameters to logger
        self.log.info('Parameters')
        #for key in self.FLAGS.__flags.keys():
        #    self.log.info('{}={}'.format(key, getattr(self.FLAGS, key)))

        # folder to save and restore model
        self.model_dir = './model/' + time.strftime('%H-%M-%d_%m-%y') + '/'
        create_dictionary(self.model_dir)

        # folders for plotting
        base_dir = './figures/' + time.strftime('%H-%M-%d_%m-%y')
        self.reward_dir = base_dir + '/Reward/'
        create_dictionary(self.reward_dir)
        self.trajectory_dir = base_dir + '/Trajectories/'
        create_dictionary(self.trajectory_dir)
        self.Qprior = base_dir + '/Qprior/'
        create_dictionary(self.Qprior)
        self.Qposterior_dir = base_dir + '/Qposterior/'
        create_dictionary(self.Qposterior_dir)
        self.policy_dir = base_dir + '/Policy/'
        create_dictionary(self.policy_dir)

        # statistics
        self.reward_training = []
        self.COM_training = []


    def train(self):
        '''
        main trainings loop
        '''

        print("Start Training")
        self.log.info('Loop over episodes')

        for self.episode in range(self.FLAGS.N_episodes):
            # statistics of episode
            rw = [] # reward
            time_env = [] # time spent interacting with environment
            time_sgd = [] # time spent on optimizing

            start = time.time()

            # loop tasks --------------------------------------------------------------------
            for n in range(self.FLAGS.N_tasks):
                # reset memory, environment, Network
                self.tempbuffer.reset()
                td_accum = 0  # td error
                state = self.env.reset()
                self.sess.run(self.Qmain.sample_prior)

                # to avoid double inference for td error calculation
                Qold = self.sess.run(self.Qmain.Qout, feed_dict={self.Qmain.state: state.reshape(-1, self.FLAGS.state_space)})[0]

                for step in range(self.FLAGS.L_episode):
                    # take a step
                    action = np.argmax(Qold)
                    next_state, reward, done, _ = self.env.step(action)

                    Qnew = self.sess.run(self.Qmain.Qout, feed_dict={self.Qmain.state: next_state.reshape(-1, self.FLAGS.state_space)})[0]

                    # store experience in memory
                    new_experience = [state, action, reward, next_state, done]
                    self.tempbuffer.add(new_experience)

                    # prioritized memory
                    factor = 1.
                    td_accum += factor * np.abs(Qold[action] - reward - self.gamma * np.max(Qnew))
                    Qold = Qnew.copy()

                    # actual reward
                    rw.append(1. * reward / self.FLAGS.rew_norm)

                    # update state, and counters
                    state = next_state.copy()

                    if done: break;

                    if (step+1) % self.FLAGS.update_freq_post == 0 and step < self.split_ratio * self.FLAGS.L_episode:
                        _, _ = self._posterior()

            time_env.append(time.time() - start)

            # append episode buffer to large buffer
            self.fullbuffer.add(td_accum, self.tempbuffer.buffer)

            reward_summary = tf.Summary(
                value=[tf.Summary.Value(tag='Training/Episodic Reward',
                                        simple_value=np.sum(np.array(rw)) / self.FLAGS.N_tasks)])
            self.summary_writer.add_summary(reward_summary, self.episode)
            self.summary_writer.flush()

            # gradient descent
            start = time.time()
            for _ in range(4):
                self._optimize()
            time_sgd.append(time.time() - start)

            # update target network
            if self.episode % self.FLAGS.update_freq_target == 0:
                vars_modelQ = self.sess.run(self.Qmain.tvars)
                feed_dict = dictionary = dict(zip(self.Qtarget.variable_holders, vars_modelQ))
                feed_dict.update({self.Qtarget.tau: self.FLAGS.tau})
                self.sess.run(self.Qtarget.copyParams, feed_dict=feed_dict)

            # save model
            if self.episode % 5000 == 0:
                self.log.info('Save model snapshot')
                filename = os.path.join(self.model_dir, 'model')
                self.saver.save(self.sess, filename, global_step=self.episode)

            # parameters
            self.batch_size = np.min([self.batch_size*2, 8])
            self.learning_rate = np.max([self.learning_rate/ self.FLAGS.lr_drop, self.FLAGS.lr_final])

            if self.noise_precision < self.FLAGS.noise_precmax and self.episode % self.FLAGS.noise_Ndrop == 0:
                self.noise_precision *= self.FLAGS.noise_precstep
            if self.episode % self.FLAGS.split_N == 0 and self.episode > 500:
                self.split_ratio = np.min([self.split_ratio + 0.003, self.FLAGS.split_ratio_max])

            # evaluation
            if self.episode % self.FLAGS.save_frequency == 0:
                print('AVG time env: ' + str(time_env))
                print('AVG time sgd: ' + str(time_sgd))

                print('Evaluation ...')
                self._evaluate()
                self._prior_rollouts()
                self._posterior_rollouts()
                self._summary()
                print('Training ...')

        # write reward to file
        df = pd.DataFrame(self.reward_training)
        df.to_csv(self.reward_dir + 'reward_per_episode', header=False, index=False, mode='a')

        # write target distance vs COM offset to file
        df = pd.DataFrame(self.COM_training)
        df.to_csv(self.reward_dir + 'target_COM', header=False, index=False, mode='a')

        #
        self._reset()


    def _evaluate(self, updates=True):
        '''
        evaluate performance on new instances w/o GD updates
        '''

        # define configurations to test
        displacement_x = 0.03
        displacement_y = (0.3+ np.array([0., 0.5, 1.])* 0.4) * self.env.maxp
        offset_EE_y = 0.01 * (-0.5 + np.array([0., 0.5, 1.]))
        offset_COM_y = 0.02 * (-1. + 2. * np.array([0., 0.5, 1.]))

        target_COM_distance = []  # distance to target
        target_EE_distance = []
        EE_COM_distance = []
        episode_length = []
        max_speed = []  # speed achieved
        speed_task = []
        rw = []
        final_y = []
        final_x = []
        termination = np.zeros(self.FLAGS.state_space)

        for dy in range(3): # displacement_y
            for ey in range(3): # offset_EE_y
                for cy in range(3): # offset_COM_y
                    # reset buffer and model
                    self.evalbuffer.reset()
                    self.sess.run(self.Qmain.sample_prior)

                    # configurations
                    state = self.env.reset(displacement_x,
                                           displacement_y[dy],
                                           offset_EE_y[ey],
                                           offset_COM_y[cy])

                    for step in range(self.FLAGS.L_episode):
                        # take a step
                        Qval = self.sess.run(self.Qmain.Qout, feed_dict={self.Qmain.state: state.reshape(-1, self.FLAGS.state_space)})[0]
                        action = np.argmax(Qval)
                        next_state, reward, done, _ = self.env.step(action)

                        # store experience in memory
                        new_experience = [state, action, reward, next_state, done]
                        self.evalbuffer.add(new_experience)

                        # actual reward
                        rw.append(1. * reward / self.FLAGS.rew_norm)
                        speed_task.append(np.linalg.norm(next_state[2:4] - state[2:4]))

                        # update state, and counters
                        state = next_state.copy()

                        if done:
                            termination += next_state < self.env.low
                            termination += next_state > self.env.high
                            break

                        #if step == 0:
                        #wt_bar, Lt = self.sess.run([self.Qmain.w0_bar, self.Qmain.L0])
                        # generate_posterior_plots(sess, QNet, wt_bar, np.linalg.inv(Lt),
                        #                         base_dir, tempbuffer, FLAGS,
                        #                         episode, step)

                        if (step + 1) % self.FLAGS.update_freq_post == 0\
                                and step < self.split_ratio * self.FLAGS.L_episode\
                                and updates:
                            wt_bar, Lt_inv = self._posterior()

                            #generate_posterior_plots(self.sess, self.Qmain, wt_bar, Lt_inv,
                            #                             self.base_dir, self.tempbuffer, self.FLAGS,
                            #                             self.episode, step)

                    target_COM_distance.append(np.linalg.norm(self.env.target_position- state[:2]- state[4:6]))
                    target_EE_distance.append(np.linalg.norm(self.env.target_position - state[:2]))
                    EE_COM_distance.append(np.linalg.norm(state[4:6]))
                    episode_length.append(step)
                    final_x.append(state[0]+ state[4])
                    final_y.append(state[1] + state[5])

        print('p_rob, v_rob, p_obj, v_obj')
        print(termination)

        # tensorflow summaries
        rw = np.sum(np.array(rw))/ len(displacement_y)/ len(offset_EE_y)/ len(offset_COM_y)
        reward_summary = tf.Summary(value=[tf.Summary.Value(tag='Validation/Episodic Reward', simple_value=rw)])
        target_COM_distance = np.mean(np.array(target_COM_distance))
        dist_COM_summary = tf.Summary(value=[tf.Summary.Value(tag='Validation/Target COM Distance', simple_value=target_COM_distance)])
        target_EE_distance = np.mean(np.array(target_EE_distance))
        dist_EE_summary = tf.Summary(
            value=[tf.Summary.Value(tag='Validation/Target EE Distance', simple_value=target_EE_distance)])
        EE_COM_distance = np.mean(np.array(EE_COM_distance))
        dist_EE_COM_summary = tf.Summary(
            value=[tf.Summary.Value(tag='Validation/EE COM Distance', simple_value=EE_COM_distance)])
        ep_len_summary = tf.Summary(
            value=[tf.Summary.Value(tag='Validation/Episode Length', simple_value=np.mean(np.array(episode_length)))])
        final_x_summary = tf.Summary(
            value=[tf.Summary.Value(tag='Validation/x position', simple_value=np.mean(np.array(final_x)))])
        final_y_summary = tf.Summary(
            value=[tf.Summary.Value(tag='Validation/y position', simple_value=np.mean(np.array(final_y)))])
        self.summary_writer.add_summary(reward_summary, self.episode)
        self.summary_writer.add_summary(dist_COM_summary, self.episode)
        self.summary_writer.add_summary(dist_EE_summary, self.episode)
        self.summary_writer.add_summary(dist_EE_COM_summary, self.episode)
        self.summary_writer.add_summary(ep_len_summary, self.episode)
        self.summary_writer.add_summary(final_x_summary, self.episode)
        self.summary_writer.add_summary(final_y_summary, self.episode)
        self.summary_writer.flush()

        '''
        # target distance vs COM offset
        position_object = np.array([state[0] + state[4], state[1] + state[5]])
        self.COM_training.append([np.linalg.norm(self.env.target_position - position_object),
                                  np.linalg.norm(self.obj_offset_COM_local)])

        # entropy of action selection
        _, action_count = np.unique(np.asarray(action_task), return_counts=True)
        action_prob = 1. * action_count / np.sum(action_count)
        entropy_episode.append(np.sum([-p * np.log(p) for p in action_prob if p != 0.]))

        # maximum speed of robot arm
        speed_task = np.asarray(speed_task)
        max_speed.append(np.max(speed_task) * self.env.control_hz)

        generate_plots(sess, summary_writer, base_dir, tempbuffer, FLAGS, episode)

        Neval = 50
        #
        reward_eval_post = 0.
        for n in range(Neval):
            reward_eval_post += evaluate_Q(QNet, evalbuffer, env, sess, FLAGS, split_ratio, noise_precision)

        reward_eval_summary = tf.Summary(
            value=[tf.Summary.Value(tag='Performance2/Eval Reward Posterior',
                                    simple_value=reward_eval_post / FLAGS.rew_norm)])
        summary_writer.add_summary(reward_eval_summary, episode)

        #
        reward_eval_prior = 0.
        for n in range(Neval):
            reward_eval_prior += evaluate_Q(QNet, evalbuffer, env, sess, FLAGS, 0., noise_precision)


        log.info('Episode %3.d with R %3.d', episode, np.sum(rw))

        # evaluation_plots(sess, QNet, env, tempbuffer, FLAGS, summary_writer, noise_precision, episode, split_ratio, base_dir)
        value_function_plot(sess, QNet, tempbuffer, FLAGS, episode, base_dir)
        policy_plot(sess, QNet, tempbuffer, FLAGS, episode, base_dir)
        '''

    def _prior_rollouts(self):
        '''
        function to plot rollouts of prior (i.e. w/o posterior updates)
        '''

        # define configurations to test
        displacement_x = 0.03
        displacement_y = 0.5 * self.env.maxp
        offset_EE_y = 0.
        offset_COM_y = 0.

        #
        buffer = []

        # rollouts
        for _ in np.arange(10):
            self.evalbuffer.reset()
            self.sess.run(self.Qmain.sample_prior)

            # configurations
            state = self.env.reset(displacement_x,
                                   displacement_y,
                                   offset_EE_y,
                                   offset_COM_y)

            for step in range(self.FLAGS.L_episode):
                # take a step
                Qval = self.sess.run(self.Qmain.Qout, feed_dict={self.Qmain.state: state.reshape(-1, self.FLAGS.state_space)})[0]
                action = np.argmax(Qval)
                next_state, reward, done, _ = self.env.step(action)

                # store experience in memory
                self.evalbuffer.add(state)

                # update state, and counters
                state = next_state.copy()

                if done:
                    break

            buffer.append(self.evalbuffer.buffer)

        # plot
        fig, ax = plt.subplots(ncols=2, figsize=[10, 4])
        ax[0].set_title('End-Effector')
        ax[1].set_title('Object')
        for num, rollout in enumerate(buffer):
            rollout = np.asarray(rollout)
            ax[0].plot(rollout[:,0], rollout[:,1], color=plt.cm.cool(num* 25))
            ax[1].plot(rollout[:,0]+ rollout[:,4],
                       rollout[:,1]+ rollout[:,5],
                       color=plt.cm.cool(num* 25))

        ax[0].set_xlim([0, 0.8])
        ax[0].set_ylim([0, 0.8])
        ax[1].set_xlim([0, 0.8])
        ax[1].set_ylim([0, 0.8])
        plt.savefig(self.policy_dir+ '/Rollout_'+ str(self.episode))
        plt.close()

        del buffer


    def _posterior_rollouts(self):
        '''
        function to plot rollouts of posterior (i.e. w/ posterior updates)
        '''

        # define configurations to test
        displacement_x = 0.03
        displacement_y = 0.5 * self.env.maxp
        offset_EE_y = 0.
        offset_COM_y = 0.


        # single rollout of prior (gather data)
        self.tempbuffer.reset()
        w0_bar = self.sess.run(self.Qmain.w0_bar)

        # configurations
        base_state = self.env.reset(displacement_x,
                                    displacement_y,
                                    offset_EE_y,
                                    offset_COM_y)

        # regular loop over trajectory
        for base_step in range(self.FLAGS.L_episode):
            Qbase = np.einsum('lm,bla->ba', w0_bar, self.sess.run(self.Qmain.phi, feed_dict={self.Qmain.state: base_state.reshape(-1, self.FLAGS.state_space)}))
            base_action = np.argmax(Qbase[0])

            base_next_state, base_reward, base_done, _ = self.env.step(base_action)

            # store experience in memory
            new_experience = [base_state, base_action, base_reward, base_next_state, base_done]
            self.tempbuffer.add(new_experience)

            base_state = base_next_state.copy()

            if base_done:
                break

            # update posterior
            if (base_step + 1) % self.FLAGS.update_freq_post == 0 and base_step < self.split_ratio* self.FLAGS.L_episode:
                # rollouts (start to end)
                buffer = []

                for _ in np.arange(10):
                    mini_buffer = []

                    #
                    self._posterior()

                    env = PushEnv()
                    env.rew_scale = self.FLAGS.rew_norm

                    # configurations
                    state = env.reset(displacement_x,
                                      displacement_y,
                                      offset_EE_y,
                                      offset_COM_y)

                    for step in range(self.FLAGS.L_episode):
                        # take a step
                        Qval = self.sess.run(self.Qmain.Qout, feed_dict={self.Qmain.state: state.reshape(-1, self.FLAGS.state_space)})[0]
                        action = np.argmax(Qval)
                        next_state, reward, done, _ = env.step(action)

                        # store experience in memory
                        mini_buffer.append(state)

                        # update state, and counters
                        state = next_state.copy()

                        if done:
                            break

                    buffer.append(mini_buffer)

                # plot
                fig, ax = plt.subplots(ncols=2, figsize=[10, 4])
                ax[0].set_title('End-Effector')
                ax[1].set_title('Object')

                for num, rollout in enumerate(buffer):
                    rollout = np.asarray(rollout)
                    ax[0].plot(rollout[:,0], rollout[:,1], color=plt.cm.cool(num* 25))
                    ax[1].plot(rollout[:,0]+ rollout[:,4],
                               rollout[:,1]+ rollout[:,5],
                               color=plt.cm.cool(num* 25))

                ax[0].set_xlim([0, 0.8])
                ax[0].set_ylim([0, 0.8])
                ax[1].set_xlim([0, 0.8])
                ax[1].set_ylim([0, 0.8])
                plt.savefig(self.policy_dir+ '/Rollout_'+ str(self.episode)+'_horizon_'+ str(base_step))
                plt.close()

        del buffer, mini_buffer


    def _reset(self):
        self.sess.close()
        self.fullbuffer.reset()
        self.tempbuffer.reset()
        self.evalbuffer.reset()

    def _posterior(self):
        length = len(self.tempbuffer.buffer)

        reward_train = np.zeros([length, ])
        state_train = np.zeros([length, self.FLAGS.state_space])
        next_state_train = np.zeros([length, self.FLAGS.state_space])
        action_train = np.zeros([length, ])
        done_train = np.zeros([length, ])

        # fill arrays
        for k, experience in enumerate(self.tempbuffer.buffer):
            # [s, a, r, s', a*, d]
            state_train[k] = experience[0]
            action_train[k] = experience[1]
            reward_train[k] = experience[2]
            next_state_train[k] = experience[3]
            done_train[k] = experience[4]

        # update
        wt_bar, Lt_inv, _ = self.sess.run([self.Qmain.wt_bar, self.Qmain.Lt_inv, self.Qmain.sample_post],
                     feed_dict={self.Qmain.context_state: state_train,
                                self.Qmain.context_action: action_train,
                                self.Qmain.context_reward: reward_train,
                                self.Qmain.context_state_next: next_state_train,
                                self.Qmain.context_done: done_train,
                                self.Qmain.nprec: self.noise_precision})

        return wt_bar, Lt_inv

    def _optimize(self):

        # to accumulate the losses across the batches
        lossBuffer = 0
        lossregBuffer = 0
        lossklBuffer = 0

        # to accumulate gradients
        gradBuffer = self.sess.run(self.Qmain.tvars)  # get shapes of tensors
        for idx in range(len(gradBuffer)):
            gradBuffer[idx] *= 0

        # Gradient descent
        for bs in range(self.batch_size):

            # sample from larger buffer [s, a, r, s', d]
            experience, index, is_weight = self.fullbuffer.sample(1)

            length = len(experience[0])

            state_sample = np.zeros((length, self.FLAGS.state_space))
            action_sample = np.zeros((length,))
            reward_sample = np.zeros((length,))
            next_state_sample = np.zeros((length, self.FLAGS.state_space))
            done_sample = np.zeros((length,))

            # fill arrays
            for k, (s0, a, r, s1, d) in enumerate(experience[0]):
                state_sample[k] = s0
                action_sample[k] = a
                reward_sample[k] = r
                next_state_sample[k] = s1
                done_sample[k] = d

            # split into context and prediction set
            split = np.int(self.split_ratio * length * np.random.rand())

            train = np.arange(0, split)
            valid = np.arange(split, length)

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
            amax_online = self.sess.run(self.Qmain.max_action,
                                       feed_dict={self.Qmain.context_state: state_train,
                                                  self.Qmain.context_action: action_train,
                                                  self.Qmain.context_reward: reward_train,
                                                  self.Qmain.context_state_next: next_state_train,
                                                  self.Qmain.state: state_valid,
                                                  self.Qmain.state_next: next_state_valid,
                                                  self.Qmain.nprec: self.noise_precision,
                                                  self.Qmain.is_online: False})

            # evaluate target model
            Qmax_target, phi_max_target = self.sess.run([self.Qtarget.Qmax,
                                                         self.Qtarget.phi_max],
                                       feed_dict={self.Qtarget.context_state: state_train,
                                                  self.Qtarget.context_action: action_train,
                                                  self.Qtarget.context_reward: reward_train,
                                                  self.Qtarget.context_state_next: next_state_train,
                                                  self.Qtarget.state: state_valid,
                                                  self.Qtarget.state_next: next_state_valid,
                                                  self.Qtarget.amax_online: amax_online,
                                                  self.Qtarget.nprec: self.noise_precision,
                                                  self.Qtarget.is_online: False})

            # update model
            grads, loss, lossreg, losskl, Qdiff = self.sess.run([self.Qmain.gradients,
                                                                 self.Qmain.loss,
                                                                 self.Qmain.loss_reg,
                                                                 self.Qmain.loss_kl,
                                                                 self.Qmain.Qdiff],
                                        feed_dict={self.Qmain.context_state: state_train,
                                                   self.Qmain.context_action: action_train,
                                                   self.Qmain.context_reward: reward_train,
                                                   self.Qmain.context_state_next: next_state_train,
                                                   self.Qmain.state: state_valid,
                                                   self.Qmain.action: action_valid,
                                                   self.Qmain.reward: reward_valid,
                                                   self.Qmain.state_next: next_state_valid,
                                                   self.Qmain.done: done_valid,
                                                   self.Qmain.amax_online: amax_online,
                                                   self.Qmain.phi_max_target: phi_max_target,
                                                   self.Qmain.Qmax_online: Qmax_target,
                                                   self.Qmain.lr_placeholder: self.learning_rate,
                                                   self.Qmain.nprec: self.noise_precision,
                                                   self.Qmain.is_online: False})

            # update prioritized replay
            # buffer.update(index[0], np.sum(np.abs(Qdiff)))
            # for idx in range(len(Qdiff)):
            self.fullbuffer.update(index[0], np.sum(np.abs(Qdiff)) * split / length)  # XXX

            for idx, grad in enumerate(grads):  # grad[0] is gradient and grad[1] the variable itself
                gradBuffer[idx] += (grad[0] / self.batch_size)

            lossBuffer += loss
            lossregBuffer += lossreg
            lossklBuffer += losskl

        # update summary
        feed_dict = dictionary = dict(zip(self.Qmain.gradient_holders, gradBuffer))
        feed_dict.update({self.Qmain.lr_placeholder: self.learning_rate})

        # reduce summary size
        if self.episode % 10 == 0:

            # update summary
            _, summaries_gradvar = self.sess.run([self.Qmain.updateModel, self.Qmain.summaries_gradvar], feed_dict=feed_dict)
            lossreg_summary = tf.Summary(value=[
                tf.Summary.Value(tag='Performance/Loss_Regularization', simple_value=(lossregBuffer / self.batch_size))])
            losskl_summary = tf.Summary(
                value=[tf.Summary.Value(tag='Performance/Loss_KLdiv', simple_value=(lossklBuffer / self.batch_size))])
            loss_summary = tf.Summary(
                value=[tf.Summary.Value(tag='Performance/Loss', simple_value=(lossBuffer / self.batch_size))])
            precision = self.sess.run(self.Qmain.L0_asym)
            max_ev_summary = tf.Summary(value=[tf.Summary.Value(tag='Parameters/Max Eigen', simple_value=np.max(precision))])
            min_ev_summary = tf.Summary(value=[tf.Summary.Value(tag='Parameters/Min Eigen', simple_value=np.min(precision))])

            self.summary_writer.add_summary(loss_summary, self.episode)
            self.summary_writer.add_summary(lossreg_summary, self.episode)
            self.summary_writer.add_summary(losskl_summary, self.episode)
            self.summary_writer.add_summary(max_ev_summary, self.episode)
            self.summary_writer.add_summary(min_ev_summary, self.episode)
            self.summary_writer.add_summary(summaries_gradvar, self.episode)

            self.summary_writer.flush()
        else:
            _ = self.sess.run([self.Qmain.updateModel], feed_dict=feed_dict)

    def _summary(self):
        learning_rate_summary = tf.Summary(
            value=[tf.Summary.Value(tag='Parameters/Learning rate', simple_value=self.learning_rate)])
        split_ratio_summary = tf.Summary(
            value=[tf.Summary.Value(tag='Parameters/Split ratio', simple_value=self.split_ratio)])
        noise_summary = tf.Summary(
            value=[tf.Summary.Value(tag='Parameters/Noise precision', simple_value=self.noise_precision)])

        self.summary_writer.add_summary(learning_rate_summary, self.episode)
        self.summary_writer.add_summary(split_ratio_summary, self.episode)
        self.summary_writer.add_summary(noise_summary, self.episode)
        self.summary_writer.flush()

    def _fill_buffer(self):
        '''
        fill replay buffer with random transitions
        '''
        for ep in range(500):
            # episode buffer
            self.tempbuffer.reset()

            # environment
            state = self.env.reset()

            step = 0
            done = False

            while (step < self.FLAGS.L_episode) and (done == False):
                # interact
                action = np.random.randint(self.FLAGS.action_space)
                next_state, reward, done, _ = self.env.step(action)

                # store experience in memory
                new_experience = [state, action, reward, next_state, done]

                # store experience in memory
                self.tempbuffer.add(new_experience)

                # update state
                state = next_state.copy()
                step += 1

            self.fullbuffer.add(1e4, self.tempbuffer.buffer)

        self.log.info('Replay Buffer Filled')

