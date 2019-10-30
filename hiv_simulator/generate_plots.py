import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import io


# Helper Functions ===========================================================
def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image

def log_dynamics(sess, writer, tag, values, step):
    """Logs the dynamics of a list/vector of values."""
    figure = plt.figure()
    plt.plot(values[:, 4:])
    plt.xlabel('time')
    plt.ylabel('Concentration')
    plt.xlim([5, 205])
    plt.ylim([-0.2, 6.2])
    plt.legend(['V', 'E'], loc='upper right')

    tf_img = plot_to_image(figure)
    summary = tf.summary.image("System Evolution", tf_img)

    writer.add_summary(sess.run(summary))
    writer.flush()

# Helper Functions ===========================================================

def generate_plots(sess, writer, base_dir, buffer, FLAGS, episode):
        trajectory_dir = base_dir + '/Trajectories/'
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

        log_dynamics(sess, writer, "System Evolution", state_train, episode)

        fig, ax = plt.subplots(ncols=5)
        for i in range(5):
            ax[i].hist(state_train[:,i])
            ax[i].set_xlabel('state '+str(i))
            ax[i].set_ylabel('count')
        plt.savefig(states_dir + 'Episode_' + str(episode))
        plt.close()

def plot_hist(ts, name, episode, folder):
    plt.figure()
    plt.hist(ts)
    plt.xlabel(name)
    plt.ylabel('count')
    plt.savefig(folder + 'Episode_' + str(episode))
    plt.close()


def log_histogram(writer, tag, values, step, bins=1000):
    """Logs the histogram of a list/vector of values."""
    # Convert to a numpy array
    values = np.array(values)

    # Create histogram using numpy
    counts, bin_edges = np.histogram(values, bins=bins)

    # Fill fields of histogram proto
    hist = tf.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values ** 2))

    # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
    # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
    # Thus, we drop the start of the first bin
    bin_edges = bin_edges[1:]

    # Add bin edges and counts
    for edge in bin_edges:
        hist.bucket_limit.append(edge)
    for c in counts:
        hist.bucket.append(c)

    # Create and write Summary
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
    writer.add_summary(summary, step)
    writer.flush()


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
        Qlog = np.zeros([Neval, FLAGS.L_episode]) # Q values
        Rlog = np.zeros([Neval, FLAGS.L_episode]) # Reward
        Vlog = np.zeros([Neval, FLAGS.L_episode]) # V marker
        Elog = np.zeros([Neval, FLAGS.L_episode]) # E marker

        Action = np.zeros([Neval, FLAGS.L_episode])
        Reward = np.zeros([Neval, FLAGS.L_episode])

        # load hidden parameters for evaluation
        preset_hidden_params = preset_parameters[i_eval]

        for ne in range(Neval):

            # reset buffer
            buffer.reset()
            # reset environment
            env.reset()
            env.param_set = preset_hidden_params
            state = env.observe()

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

                if FLAGS.rew_log:
                    rew_log = np.exp(reward * FLAGS.rew_norm)
                else:
                    rew_log = reward * FLAGS.rew_norm

                # logging
                Qlog[ne, step] = np.max(Qval)
                Rlog[ne, step] = reward
                Vlog[ne, step] = state[4]
                Elog[ne, step] = state[5]

                Action[ne, step] = action
                Reward[ne, step] = rew_log

                if done == 1:
                    break

                # update state, and counters
                state = next_state.copy()
                step += 1


        #generate_posterior_plots(sess, QNet, base_dir, episode, i_eval, 0)
        policy_plot(sess, QNet, buffer, FLAGS, episode, i_eval, 0, base_dir)
        # -----------------------------------------------------------------------

        reward_tensorboard = np.mean(np.sum(Reward, axis=1))

        # log to tensorboard
        reward_eval_summary = tf.Summary(
            value=[tf.Summary.Value(tag='Performance2/Reward Eval ' + str(i_eval), simple_value=reward_tensorboard)])
        summary_writer.add_summary(reward_eval_summary, episode)
        summary_writer.flush()

        log_histogram(summary_writer, 'Histogram/Action patient '+str(i_eval), Action.reshape(-1), episode, bins=20)
        log_histogram(summary_writer, 'Histogram/Reward patient '+str(i_eval), Reward.reshape(-1), episode, bins=20)

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


def generate_mesh(Nx=30, Ny=20):
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
    T1vec = T1 * np.ones([Nx * Ny, 1])
    T2vec = T2 * np.ones([Nx * Ny, 1])
    T1_starvec = T1_star * np.ones([Nx * Ny, 1])
    T2_starvec = T2_star * np.ones([Nx * Ny, 1])
    mesh = np.concatenate([T1vec,
                           T2vec,
                           T1_starvec,
                           T2_starvec,
                           Emesh.reshape(-1, 1),
                           Vmesh.reshape(-1, 1)], axis=1)

    return mesh


def generate_posterior_plots(sess, QNet, base_dir, episode, patient, step):
    '''
    plot cut through value function
    '''
    Qposterior_dir = base_dir + '/Qposterior/'
    Nx = 30
    Ny = 20

    mesh = generate_mesh()

    w, Linv, phi, = sess.run([QNet.wt_bar, QNet.Lt_inv, QNet.phi], feed_dict={QNet.state: mesh})

    Q = np.einsum('im,bia->ba', w, phi)
    dQ = np.einsum('bia,ij,bja->ba', phi, Linv, phi)

    argmax_Q = np.argmax(Q, axis=1)
    Vpost = Q[np.arange(len(Q)), argmax_Q]
    dVpost = dQ[np.arange(len(Q)), argmax_Q]

    fig, ax = plt.subplots(figsize=[8,4], ncols=2)
    im = ax[0].imshow(Vpost.reshape(Ny, Nx), extent=[0, 8, 0, 4])  # (y, x)
    ax[0].set_title('Value mean')
    ax[0].set_xlabel('V')
    ax[0].set_ylabel('E')
    fig.colorbar(im, ax=ax[0], orientation="horizontal", pad=0.2)
    im = ax[1].imshow(dVpost.reshape(Ny, Nx), extent=[0, 8, 0, 4])  # (y, x)
    ax[1].set_title('Value std')
    ax[1].set_xlabel('V')
    ax[1].set_ylabel('E')
    fig.colorbar(im, ax=ax[1], orientation="horizontal", pad=0.2)
    plt.savefig(Qposterior_dir + 'Episode_' + str(episode)+ '_patient_'+ str(patient)+ '_step_'+ str(step))
    plt.close()


def policy_plot(sess, QNet, buffer, FLAGS, episode, patient, step, base_dir):
    '''
    plot cut through policy
    '''
    policy_dir = base_dir + '/Policy/'
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

    mesh = generate_mesh()

    #
    Qmean = sess.run(QNet.Qout, feed_dict={QNet.state: mesh})
    Policy = np.argmax(Qmean, axis=1)

    Policy[0:4] = np.arange(4)

    plt.figure()
    im = plt.imshow(Policy.reshape(Ny, Nx), extent=[0,8,0,4]) # (y, x)
    plt.scatter(state_train[:, 4], state_train[:, 5])
    plt.xlabel('V')
    plt.ylabel('E')
    plt.xlim([0, 8])
    plt.ylim([0, 4])
    plt.colorbar(im, boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5])
    plt.savefig(policy_dir+ 'Episode_'+ str(episode)+ '_Patient_'+ str(patient) + '_Step_'+ str(step))
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

    mesh = generate_mesh()

    Qprior = sess.run(QNet.Qout, feed_dict={QNet.state: mesh})
    Vprior = np.max(Qprior, axis=1)

    plt.figure()
    plt.imshow(Vprior.reshape(Ny, Nx), extent=[0,8,0,4]) # (y, x)
    plt.scatter(state_train[:, 4], state_train[:, 5])
    plt.xlabel('V')
    plt.ylabel('E')
    plt.savefig(Qprior_dir+ 'Episode_'+ str(episode))
    plt.close()