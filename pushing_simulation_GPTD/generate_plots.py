import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import pickle
import tensorflow as tf
import io

maxp = 0.4

# Helper Functions Tensorflow ================================================
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

# Helper Functions ===========================================================

def plot_hist(ts, name, episode, folder):
    plt.figure()
    plt.hist(ts)
    plt.xlabel(name)
    plt.ylabel('count')
    plt.savefig(folder + 'Episode_' + str(episode))
    plt.close()


def generate_mesh(Nx=30, Ny=20):
    #
    vx = 0.
    vy = 0.
    friction = 0.12

    # loop over x,y of state
    px_robot = np.linspace(0., maxp, Nx)
    py_robot = np.linspace(0., maxp, Ny)

    px_robot_mesh, py_robot_mesh = np.meshgrid(px_robot, py_robot)

    vx_robot = vx * np.ones([Nx * Ny, 1])
    vy_robot = vy * np.ones([Nx * Ny, 1])

    px_object = px_robot_mesh+ 0.02 # ~ radius
    py_object = py_robot_mesh

    mesh = np.concatenate([px_robot_mesh.reshape(-1, 1),
                           py_robot_mesh.reshape(-1, 1),
                           vx_robot.reshape(-1, 1),
                           vy_robot.reshape(-1, 1),
                           px_object.reshape(-1, 1),
                           py_object.reshape(-1, 1)], axis=1)

    return mesh

# Plot Functions ==========================================================
def generate_plots(sess, writer, base_dir, buffer, FLAGS, episode):
    trajectory_dir = base_dir + '/Trajectories/'
    trajectory_length = len(buffer.buffer)

    reward_train = np.zeros([trajectory_length, ])
    state_train = np.zeros([trajectory_length, FLAGS.state_space])
    next_state_train = np.zeros([trajectory_length, FLAGS.state_space])
    action_train = np.zeros([trajectory_length, ])
    exponent_train = np.zeros([trajectory_length, ])
    done_train = np.zeros([trajectory_length, 1])

    # fill arrays
    for k, experience in enumerate(buffer.buffer):
        # [s, a, r, s', a*, d]
        state_train[k] = experience[0]
        action_train[k] = experience[1]
        reward_train[k] = experience[2]
        next_state_train[k] = experience[3]
        exponent_train[k] = experience[4]
        done_train[k] = experience[5]


    fig, ax = plt.subplots()
    ax.scatter(state_train[:, 0], state_train[:, 1], color='b', marker='o')
    ax.scatter(state_train[:, 4], state_train[:, 5], color='b', marker='s')
    rect = mp.Rectangle((0, 0), maxp, maxp, edgecolor='b', linewidth=1, facecolor='none')
    ax.add_patch(rect)
    ax.scatter(0.8* maxp, 0.5* maxp, color='r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.savefig(trajectory_dir + 'Episode_' + str(episode))
    plt.close()


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


def policy_plot(sess, QNet, buffer, FLAGS, episode, base_dir):
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
    exponent_train = np.zeros([trajectory_length, ])
    done_train = np.zeros([trajectory_length, 1])

    # fill arrays
    for k, experience in enumerate(buffer.buffer):
        # [s, a, r, s', a*, d]
        state_train[k] = experience[0]
        action_train[k] = experience[1]
        reward_train[k] = experience[2]
        next_state_train[k] = experience[3]
        exponent_train[k] = experience[4]
        done_train[k] = experience[5]

    mesh = generate_mesh()

    #
    Qmean = sess.run(QNet.Qout, feed_dict={QNet.state: mesh})
    Policy = np.argmax(Qmean, axis=1)

    Policy[0:5] = np.arange(5)

    plt.figure()
    im = plt.imshow(Policy.reshape(Ny, Nx), extent=[0, maxp, 0., maxp]) # (y, x)
    plt.scatter(state_train[:, 4], state_train[:, 5])
    plt.xlabel('V')
    plt.ylabel('E')
    plt.xlim([0, maxp])
    plt.ylim([0, maxp])
    plt.colorbar(im, boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    plt.savefig(policy_dir+ 'Episode_'+ str(episode))
    plt.close()


def value_function_plot(sess, QNet, buffer, FLAGS, episode, base_dir):
    '''
    plot cut through value function
    '''
    Qprior_dir = base_dir + '/Qprior/'
    Nx = 30
    Ny = 20

    mesh = generate_mesh()

    Qprior = sess.run(QNet.Qout, feed_dict={QNet.state: mesh})
    Vprior = np.max(Qprior, axis=1)

    plt.figure()
    plt.imshow(Vprior.reshape(Ny, Nx), extent=[0, maxp, 0, maxp]) # (y, x)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(Qprior_dir+ 'Episode_'+ str(episode))
    plt.close()