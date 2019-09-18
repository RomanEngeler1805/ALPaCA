import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from QNetwork import QNetwork
from synthetic_data_sampler import sample_wheel_bandit_data
from evaluation_tf import Qeval
import sys
import time
import os

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

tf.flags.DEFINE_integer("L_episode", 500, "Length of episodes")

tf.flags.DEFINE_integer("split_N", 30, "Increase split ratio every N steps")
tf.flags.DEFINE_float("split_ratio", 0.1, "Initial split ratio for conditioning")
tf.flags.DEFINE_integer("update_freq_post", 1, "Update frequency of posterior and sampling of new policy")

tf.flags.DEFINE_integer("replay_memory_size", 1000, "Size of replay memory")
tf.flags.DEFINE_integer("iter_amax", 1, "Number of iterations performed to determine amax")
tf.flags.DEFINE_integer("save_frequency", 400, "Store images every N-th episode")
tf.flags.DEFINE_float("regularizer", 1.0, "Regularization parameter")
tf.flags.DEFINE_string('non_linearity', 'relu', 'Non-linearity used in encoder')

tf.flags.DEFINE_integer("random_seed", 2345, "Random seed for numpy and tensorflow")

tf.flags.DEFINE_string('model_name', '14-22-31_08-19', 'Name of the model for evaluation')

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)

from wheel_bandit_environment import wheel_bandit_environment
env = wheel_bandit_environment(FLAGS.action_space, FLAGS.random_seed)

# load tf model
load_dir = './model/'+ FLAGS.model_name+ '/'
save_dir = './eval/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#
# sample dataset
num_contexts = 80000
num_datasets = 100
batch_size = 1000

num_actions = 5
context_dim = 2
mean_v = [1.2, 1.0, 1.0, 1.0, 1.0]
std_v = [0.01, 0.01, 0.01, 0.01, 0.01]
mu_large = 50
std_large = 0.01

# Create Session
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.12)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    QNet = QNetwork(FLAGS)
    Qeval = Qeval(FLAGS)

    saver = tf.train.Saver(max_to_keep=4)
    #saver.restore(sess, tf.train.latest_checkpoint(load_dir))
    saver.restore(sess, load_dir+'model-2800')
    print('Successfully restored model from '+ str(tf.train.latest_checkpoint(load_dir)))

    # Loop over deltas (exploration parameter)
    for delta in [0.5, 0.7, 0.9, 0.95, 0.99]:

        # np.hstack((contexts, rewards)), (opt_rewards, opt_actions)
        dataset, opt_wheel = sample_wheel_bandit_data(num_contexts, delta, mean_v, std_v, mu_large, std_large)

        # monitor time
        start = time.time()

        # obtain encoding
        phi = np.zeros([num_contexts, FLAGS.latent_space, FLAGS.action_space])
        for i in range(0, num_contexts, batch_size):
            phi[i:i+ batch_size] = sess.run(QNet.phi, feed_dict={QNet.state: dataset[i: i+batch_size, :2]})

        # arrays for final statistics
        rew_agent_episode_cumulative = np.zeros([num_datasets])
        rew_rand_episode_cumulative = np.zeros([num_datasets])

        rew_agent_episode_simple = np.zeros([num_datasets])
        rew_rand_episode_simple = np.zeros([num_datasets])
        rew_opt_episode_simple = np.zeros([num_datasets])

        # array to keep track of indices of shuffled array
        data_idx = np.arange(num_contexts)

        # 50 random ordered datasets
        for shuffle in range(num_datasets):
            #if shuffle% 10 == 0:
                #print('Iteration {:2d}...'.format(shuffle))
            # shuffle dataset
            np.random.shuffle(data_idx)

            # array
            actions = np.zeros(num_contexts)
            rew_agent_online = np.zeros(num_contexts)
            rew_rand_online = np.zeros(num_contexts)

            # sample prior (reset for new evaluation)
            sess.run(QNet.reset_post)
            sess.run(Qeval.sample_prior)

            # loop over dataset
            for i in range(np.int(FLAGS.split_ratio* FLAGS.L_episode)):

                # calculate posterior
                sess.run(Qeval.sample_post,
                            feed_dict={Qeval.context_phi: phi[data_idx[:i]],
                                    Qeval.context_action: actions[:i],
                                    Qeval.context_reward: rew_agent_online[:i],
                                    Qeval.nprec: FLAGS.noise_precision,
                                    Qeval.is_online: True})

                # prediction
                Qval = sess.run(Qeval.Qout, feed_dict={Qeval.phi: phi[data_idx[i]].reshape(-1, FLAGS.latent_space, FLAGS.action_space)})
                action = np.argmax(Qval, axis=1)

                actions[i] = action
                rew_agent_online[i] = dataset[data_idx[i], 2+ action]
                rew_rand_online[i] = dataset[data_idx[i], 2 + np.random.randint(0,5)]

            for i in range(np.int(FLAGS.split_ratio* FLAGS.L_episode), num_contexts, batch_size):
                # prediction
                Qval = sess.run(Qeval.Qout, feed_dict={Qeval.phi: phi[data_idx[i:i+batch_size]].reshape(-1, FLAGS.latent_space, FLAGS.action_space)})
                action = np.argmax(Qval, axis=1)

                actions[i:i+batch_size] = action
                rew_agent_online[i:i+batch_size] = dataset[data_idx[i:i+batch_size], 2+ action]
                rew_rand_online[i:i+batch_size] = dataset[data_idx[i:i+batch_size], 2 + np.random.randint(0, 5, size=np.min([batch_size, num_contexts-i]))]

            # store observed reward
            rew_agent_episode_cumulative[shuffle] = np.sum(rew_agent_online)
            rew_rand_episode_cumulative[shuffle] = np.sum(rew_rand_online)

            rew_agent_episode_simple[shuffle] = np.sum(rew_agent_online[-500:])
            rew_rand_episode_simple[shuffle] = np.sum(rew_rand_online[-500:])
            rew_opt_episode_simple[shuffle] = np.sum(opt_wheel[data_idx,0][-500:])

        # print reward
        # cumulative regret
        mean_reward_agent = np.mean(rew_agent_episode_cumulative)
        stdv_reward_agent = np.std(rew_agent_episode_cumulative)
        mean_reward_rand = np.mean(rew_rand_episode_cumulative)
        mean_reward_opt = np.sum(opt_wheel[:,0])

        cum_regret = mean_reward_opt- mean_reward_agent
        cum_regret_norm = 100.* cum_regret/ (mean_reward_opt-  mean_reward_rand)
        dcum_regret_norm = 100./np.sqrt(num_datasets)* stdv_reward_agent/ (mean_reward_opt- mean_reward_rand)

        # simple regret
        smean_reward_agent = np.mean(rew_agent_episode_simple)
        sstdv_reward_agent = np.std(rew_agent_episode_simple)
        smean_reward_rand = np.mean(rew_rand_episode_simple)
        smean_reward_opt = np.mean(rew_opt_episode_simple)

        simple_regret = smean_reward_opt- smean_reward_agent
        simple_regret_norm = 100. * simple_regret / (smean_reward_opt - smean_reward_rand)
        dsimple_regret_norm = 100. / np.sqrt(num_datasets) * sstdv_reward_agent / (smean_reward_opt - smean_reward_rand)


        file = open(save_dir+ FLAGS.model_name, 'a')
        file.write('===================================\n')
        file.write('=========== delta = '+ str(delta) +' ==========\n')
        file.write('Reward LSTD: \n')
        file.write('{:6.2f} +- {:5.2f} ({:2.0f} %)\n'.format(mean_reward_agent,
                                            stdv_reward_agent,
                                            stdv_reward_agent/ mean_reward_agent* 100.))

        file.write('-----------------------------------\n')
        file.write('Random Reward: \n')
        file.write('{:6}\n'.format(mean_reward_rand))

        file.write('-----------------------------------\n')
        file.write('Optimal Reward: \n')
        file.write('{:6}\n'.format(mean_reward_opt))

        file.write('-----------------------------------\n')
        file.write('Cumulative Regret: \n')
        file.write('{:3.2f}% +- {:2.2f}%\n'.format(cum_regret_norm, dcum_regret_norm))

        file.write('Simple Regret: \n')
        file.write('{:3.2f}% +- {:2.2f}%\n'.format(simple_regret_norm, dsimple_regret_norm))

        file.write('-----------------------------------\n')
        file.write('Time for evaluation: {:5f}\n'.format(time.time() - start))
        file.write('===================================\n')

        file.close()
