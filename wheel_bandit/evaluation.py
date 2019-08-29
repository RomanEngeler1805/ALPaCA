import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from QNetwork import QNetwork
from synthetic_data_sampler import sample_wheel_bandit_data
from evaluation_tf import Qeval

# for deltas in NP paper
#   for i in {0, .. 50}
#       buff <- draw 80k samples
#       loop over buff
#           predict
#           update (posterior) with observations
#   save rewards/ regrets

# implement incremental GP updates!!!!

class Bayesian_regression():
    def __init__(self, w0, L0, num_instances):
        self.num_instances = num_instances

        # prior
        self.w0 = w0
        self.L0 = L0

        self.Sigma_e = 1e-3

        self.wt = self.w0

        # posterior (parallelized)
        self.wt_bar = np.zeros([self.num_instances, self.w0.shape])
        self.Lt_inv = np.zeros([self.num_instances, self.L0.shape[0], self.L0.shape[1]])

    def update_posterior(self, phit, rt):
        if not phit.shape[1] == self.num_instances:
            raise ERROR

        # inverse precision
        denum = 1+ np.einsum('bi,bij,bj->b', phit, self.Lt_inv, phit)
        denum = np.reciprocal(denum)

        num = np.einsum('bij,bj->bi', self.Lt_inv, phit)
        num = np.einsum('bi,bj->bij', num, num)

        self.Lt_inv = self.Lt_inv- np.einsum('bij,b->bij', num, denum)

        # mean
        self.qt = self.qt+ np.einsum('bi,b->bi', phit, rt)
        self.wt_bar = np.einsum('ij,bj->bi', self.Lt_inv, self.qt)

    def predict(self, phit):
        # predictive
        yt = tf.einsum('i,bi->b',self.wt, phit)
        Sigma_t = self.Sigma_e* (1.+ tf.einsum('bi,ij,bj->b', phit, self.Lt_inv, phit))

        return yt, Sigma_t



# load tf model
model_dir = ''

with tf.Session() as sess:
    QNet = QNetwork()

    saver = tf.train.Saver(max_to_keep=4)
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))
    print('Successully restored model from '+ str(tf.train.latest_checkpoint(model_dir)))

# sample dataset
num_contexts = 80000
num_datasets = 50

num_actions = 5
context_dim = 2
mean_v = [1.0, 1.0, 1.0, 1.0, 1.2]
std_v = [0.01, 0.01, 0.01, 0.01, 0.01]
mu_large = 50
std_large = 0.01

noise_precision = 10.

for delta in [0.5, 0.7, 0.9, 0.95]:
    dataset, opt_wheel = sample_wheel_bandit_data(num_contexts, delta, mean_v, std_v, mu_large, std_large)

# obtain encoding
    phi = np.zeros([num_contexts, latent_dim])
    batch_size = 100

    for i in range(0, num_contexts, batch_size):
        phi[i:i+ batch_size] = sess.run(QNet.phi, feed_dict={QNet.state: dataset[i: i+batch_size]})

# store encoding

# load encoding
    # 50 random ordered datasets
    data_idx = np.zeros([num_datasets, num_contexts])
    for shuffle in range(num_datasets):
        # shuffle dataset
        data_idx[shuffle] = np.random.permutation(num_contexts)

    # array
    actions = np.zeros(num_datasets, num_contexts)
    rewards = np.zeros(num_datasets, num_contexts)
    dones = np.zeros(num_datasets, num_contexts)


    # loop over dataset
    for i in range(num_contexts):
        # calculate posterior
        if i < 500: #the context size used in training
            print(dataset.shape)
            print(data_idx.shape)
            print(dataset[data_idx[:,:i]].shape)
            sess.run(QNet.sample_post,
                     feed_dict={QNet.context_state: dataset[data_idx[:,:i]],
                                QNet.context_action: actions[data_idx[:,:i]],
                                Net.context_reward: rewards[data_idx[:,:i]],
                                QNet.context_state_next: dataset[data_idx[:,:i]],
                                QNet.context_done: dones[data_idx[:,:i]],
                                QNet.nprec: noise_precision})

        # prediction
        Qval = sess.run([QNet.Qout], feed_dict={QNet.phi: dataset[data_idx[:,i]]})
        action = np.argmax(Qval, axis=1)
        print(action.shape)
        next_state, reward, done = [set[(step + 1) % L_valid, :2], mu[action], 0]  # cartesian

# store reward

# uniform, optimal, Thompson w.r.t. random sample from the Gaussians (synthetic_data_sampler)

# Q: what's the optimal context size?