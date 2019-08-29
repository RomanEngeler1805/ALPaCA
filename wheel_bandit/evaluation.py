import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from QNetwork import QNetwork
from synthetic_data_sampler import sample_wheel_bandit_data
from evaluation_tf import Qeval
import sys
import time

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
tf.flags.DEFINE_integer("update_freq_post", 1, "Update frequency of posterior and sampling of new policy")

tf.flags.DEFINE_integer("replay_memory_size", 1000, "Size of replay memory")
tf.flags.DEFINE_integer("iter_amax", 1, "Number of iterations performed to determine amax")
tf.flags.DEFINE_integer("save_frequency", 400, "Store images every N-th episode")
tf.flags.DEFINE_float("regularizer", 0.01, "Regularization parameter")
tf.flags.DEFINE_string('non_linearity', 'relu', 'Non-linearity used in encoder')

tf.flags.DEFINE_integer("random_seed", 1234, "Random seed for numpy and tensorflow")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)

from wheel_bandit_environment import wheel_bandit_environment
env = wheel_bandit_environment(FLAGS.action_space, FLAGS.random_seed)

# for deltas in NP paper
#   for i in {0, .. 50}
#       buff <- draw 80k samples
#       loop over buff
#           predict
#           update (posterior) with observations
#   save rewards/ regrets

# implement incremental GP updates!!!!

# load tf model
model_dir = './model/16-58-27_08-19/'

#
# sample dataset
num_contexts = 80000
num_cond = 512
num_datasets = 2
batch_size = 1000

num_actions = 5
context_dim = 2
mean_v = [1.2, 1.0, 1.0, 1.0, 1.0]
std_v = [0.01, 0.01, 0.01, 0.01, 0.01]
mu_large = 50
std_large = 0.01

noise_precision = 10.

delta = 0.7

with tf.Session() as sess:
    QNet = QNetwork(FLAGS)
    Qeval = Qeval(FLAGS)

    saver = tf.train.Saver(max_to_keep=4)
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))
    print('Successully restored model from '+ str(tf.train.latest_checkpoint(model_dir)))

    # np.hstack((contexts, rewards)), (opt_rewards, opt_actions)
    dataset, opt_wheel = sample_wheel_bandit_data(num_contexts, delta, mean_v, std_v, mu_large, std_large)

    # obtain encoding
    phi = np.zeros([num_contexts, FLAGS.latent_space, FLAGS.action_space])

    start = time.time()

    for i in range(0, num_contexts, batch_size):
        phi[i:i+ batch_size] = sess.run(QNet.phi, feed_dict={QNet.state: dataset[i: i+batch_size, :2]})

    # store encoding?

    # load encoding?

    #
    rew_agent = np.zeros([num_datasets])

    #
    data_idx = np.arange(num_contexts)

    # 50 random ordered datasets
    for shuffle in range(num_datasets):
        print('Iteration {:2d}...'.format(shuffle))
        # shuffle dataset
        np.random.shuffle(data_idx)

        # array
        actions = np.zeros(num_contexts)
        rewards = np.zeros(num_contexts)

        # sample prior
        sess.run(QNet.sample_prior)

        # loop over dataset
        for i in range(num_cond):

            # calculate posterior
            sess.run(Qeval.sample_post,
                        feed_dict={Qeval.context_phi: phi[data_idx[:i]],
                                Qeval.context_action: actions[:i],
                                Qeval.context_reward: rewards[:i],
                                Qeval.nprec: noise_precision})

            # prediction
            Qval = sess.run(Qeval.Qout, feed_dict={Qeval.phi: phi[data_idx[i]].reshape(-1, FLAGS.latent_space, FLAGS.action_space)})
            action = np.argmax(Qval, axis=1)

            actions[i] = action
            rew_agent[shuffle] += dataset[data_idx[i], 2+ action] # first two entries are state

        for i in range(num_cond, num_contexts, batch_size):
            # prediction
            Qval = sess.run(Qeval.Qout, feed_dict={Qeval.phi: phi[data_idx[i:i+batch_size]].reshape(-1, FLAGS.latent_space, FLAGS.action_space)})
            action = np.argmax(Qval, axis=1)

            actions[i:i+batch_size] = action
            rew_agent[shuffle] += np.sum(dataset[data_idx[i:i+batch_size], 2+ action]) # first two entries are state

    # print reward
    print('===================================')
    print('Reward LSTD: ')
    print('{:6.2f} +- {:5.2f} ({:2.0f} %)'.format(np.mean(rew_agent),
                                        np.std(rew_agent),
                                        np.std(rew_agent)/np.mean(rew_agent)* 100.))

    print('Time for evaluation: {:5f}'.format(time.time() - start))
    #print('Frequency of actions: ')
    #print(['({:1d}: {:6d}) '.format(idx, a) for idx, a in enumerate(actions)])

    print('===================================')
    print('Optimal Reward: ')
    print('{:6}'.format(np.sum(opt_wheel[0])))

# uniform, optimal, Thompson w.r.t. random sample from the Gaussians (synthetic_data_sampler)

# Q: what's the optimal context size?