import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf

def generate_plots(base_dir, buffer, FLAGS, episode):
        trajectory_dir = base_dir + '/Trajectories/'
        histogram_dir = base_dir + '/Action_Histogram/'
        reward_dir = base_dir + '/Reward_Histogram/'
        states_dir = base_dir + '/State_Histogram/'

        trajectory_length = len(buffer.buffer)

        reward_train = np.zeros([trajectory_length, ])
        state_train = np.zeros([trajectory_length, FLAGS.state_space])
        next_state_train = np.zeros([trajectory_length, FLAGS.state_space])
        action_train = np.zeros([trajectory_length, ])
        done_train = np.zeros([trajectory_length, 1])

        # fill arrays
        for k, experience in enumerate(buffer.buffer):
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
        plt.savefig(trajectory_dir+ 'Episode_'+ str(episode))
        plt.close()

        plt.figure()
        plt.hist(action_train)
        plt.xlabel('action')
        plt.ylabel('count')
        plt.savefig(histogram_dir + 'Episode_' + str(episode))
        plt.close()

        plt.figure()
        plt.hist(reward_train)
        plt.xlabel('reward')
        plt.ylabel('count')
        plt.savefig(reward_dir + 'Episode_' + str(episode))
        plt.close()

        fig, ax = plt.subplots(ncols=5)
        for i in range(5):
            ax[i].hist(state_train[:,i])
            ax[i].set_xlabel('state '+str(i))
            ax[i].set_ylabel('count')
        plt.savefig(states_dir + 'Episode_' + str(episode))
        plt.close()


def evaluation_plots(sess,
                     QNet,
                     env,
                     buffer,
                     FLAGS,
                     summary_writer,
                     noise_precision,
                     episode,
                     split_ratio,
                     base_dir):
    # ==================================================== #
    # log history of Q values, history of discounted rewards -> Qlog, Rlog
    # log V, E markers -> Vlog, Elog
    # log reward -> Rlog
    # ==================================================== #
    Q_Sum_r_dir = base_dir + '/Q_Sum_r/'
    V_E_dir = base_dir + '/Marker_V_E/'

    Neval = 5  # number of repeat (to get statistics)

    # evaluation ------------------------------------------------
    with open('./' + 'hiv' + '_preset_hidden_params', 'r') as f:
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
            buffer.reset()
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
                action = np.argmax(Qval)
                next_state, reward, done = env.step(action, perturb_params=True)

                # store experience in memory
                new_experience = [state, action, reward, next_state, done]
                buffer.add(new_experience)

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

        reward_tensorboard = np.mean(np.sum(Rlog, axis=1)) * FLAGS.rew_norm

        # log to tensorboard
        reward_eval_summary = tf.Summary(
            value=[tf.Summary.Value(tag='Reward Eval ' + str(i_eval), simple_value=reward_tensorboard)])
        summary_writer.add_summary(reward_eval_summary, episode)
        summary_writer.flush()

        # plot for visual inspection -----------------------------------------------
        discounted_r = np.zeros_like(Rlog[0], dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(Rlog[0]))):
            running_add = running_add * FLAGS.gamma + Rlog[0, t]
            discounted_r[t] = running_add

        # Q vs discounted summed reward
        plt.figure()
        plt.scatter(discounted_r, Qlog[0])
        plt.xlabel('summed discounted reward')
        plt.ylabel('Q value')
        plt.savefig(Q_Sum_r_dir + 'Episode_' + str(episode) + '_case_' + str(i_eval))
        plt.close()

        # V vs E
        plt.figure()
        plt.scatter(Vlog[0], Elog[0])
        plt.xlabel('V')
        plt.ylabel('E')
        plt.xlim([-1, 10])
        plt.ylim([-1, 10])
        plt.savefig(V_E_dir + 'Episode_' + str(episode) + '_case_' + str(i_eval))
        plt.close()


def value_function_plot(sess, QNet, buffer, FLAGS, episode, base_dir):
    '''
    plot cut through value function
    '''
    Qprior_dir = base_dir + '/Qprior/'
    Nx = 30
    Ny = 20

    # visualize observed states
    trajectory_length = len(buffer.buffer)
    reward_train = np.zeros([trajectory_length, ])
    state_train = np.zeros([trajectory_length, FLAGS.state_space])
    next_state_train = np.zeros([trajectory_length, FLAGS.state_space])
    action_train = np.zeros([trajectory_length, ])
    done_train = np.zeros([trajectory_length, 1])

    # fill arrays
    for k, experience in enumerate(buffer.buffer):
        # [s, a, r, s', a*, d]
        state_train[k] = experience[0]
        action_train[k] = experience[1]
        reward_train[k] = experience[2]
        next_state_train[k] = experience[3]
        done_train[k] = experience[4]

    # physical values for states
    # ("T1", "T2", "T1*", "T2*", "V", "E")
    T1 = 5.5
    T2 = 2.
    T1_star = 2.5
    T2_star = 1.0

    # loop over E [0, 4], V [0, 8] of state
    V = np.linspace(0., 8., Nx)
    E = np.linspace(0., 4., Ny)

    Vmesh, Emesh = np.meshgrid(V, E)
    T1vec = T1* np.ones([Nx* Ny, 1])
    T2vec = T2* np.ones([Nx* Ny, 1])
    T1_starvec = T1_star* np.ones([Nx* Ny, 1])
    T2_starvec = T2_star* np.ones([Nx* Ny, 1])
    mesh = np.concatenate([T1vec,
                           T2vec,
                           T1_starvec,
                           T2_starvec,
                           Emesh.reshape(-1, 1),
                           Vmesh.reshape(-1, 1)], axis=1)

    Qprior = sess.run(QNet.Q0, feed_dict={QNet.state: mesh})
    Qprior = np.max(Qprior, axis=1)

    plt.figure()
    plt.imshow(Qprior.reshape(Ny, Nx), extent=[0,8,0,4]) # (y, x)
    plt.scatter(state_train[:, 4], state_train[:, 5]) # TODO make the orientation sure of this!!!!
    plt.xlabel('V')
    plt.ylabel('E')
    plt.savefig(Qprior_dir+ 'Episode_'+ str(episode))
    plt.close()