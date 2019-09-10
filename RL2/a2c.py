import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from utils import *
import time

class A2C:
    def __init__(self,
        env,
        session,
        scope,
        policy_cls,
        hidden_dim=256,
        action_dim=3,
        encode_state=False,
        grad_clip=10):

        # parameters
        self.input_dim = 9
        self.action_dim = 1
        self.vf_ceof = 0.05
        self.ent_coef = 0.01
        self.step_size = 2e-4
        self.session = session

        #TODO: make sure to use encode_state when moving to vision domains

        # inputs
        self.X = tf.placeholder(tf.float32, [None, self.input_dim])
        self.ADV = tf.placeholder(tf.float32, [None])
        self.A = tf.placeholder(tf.int32,   [None])
        self.R = tf.placeholder(tf.float32, [None])

        self.Aold = tf.placeholder(tf.int32,   [None])
        self.Rold = tf.placeholder(tf.float32,   [None])

        # policy network
        input = tf.concat([self.X, tf.one_hot(self.Aold, self.action_dim), tf.reshape(self.Rold, [-1,1])],axis=1)
        #input = tf.contrib.layers.fully_connected(input, 64, activation_fn=tf.nn.relu)
        self.policy = policy_cls(scope=scope, inputs=input,
                                 action_dim=action_dim, hidden_dim=hidden_dim)

        # loss
        neglogpi = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.policy.pi, labels=self.A)

        # policy gradient loss?
        self.pg_loss = tf.reduce_mean(self.ADV * neglogpi)
        # value function loss?
        self.vf_loss = tf.reduce_mean(tf.square(tf.squeeze(self.policy.V) - self.R) / 2.)

        a0 = self.policy.pi - tf.reduce_max(self.policy.pi, 1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, 1, keepdims=True)
        p0 = ea0 / z0
        self.entropy = tf.reduce_mean(tf.reduce_sum(p0 * (tf.log(z0) - a0), 1))

        self.loss = self.pg_loss - self.ent_coef * self.entropy + self.vf_ceof * self.vf_loss

        # action selection
        rand_u = tf.random_uniform(tf.shape(self.policy.pi)) # introduce randomness in policy
        self.act = tf.squeeze(tf.argmax(self.policy.pi - tf.log(-tf.log(rand_u)), axis=-1))

        # optimizer
        opt = tf.train.RMSPropOptimizer(learning_rate=self.step_size, decay=0.99, epsilon=1e-5)
        self.train_op = minimize_and_clip(opt, self.loss, var_list=self.policy.variables, clip_val=grad_clip)

        # summary writer
        self.summary_writer = tf.summary.FileWriter(
            logdir=os.path.join('./', 'summaries/', time.strftime('%H-%M-%d_%m-%y')),
            graph=self.session.graph)

        initialize(self.session)

    def get_actions(self, X, A, R):
        if not self.policy.recurrent:
            actions, values = self.session.run([self.act, self.policy.V], feed_dict={self.X: X})
        else:
            print(X.shape)
            actions, values, h_out = self.session.run([self.act, self.policy.V, self.policy.h_out],
                feed_dict={self.X: X, self.Aold: A, self.Rold: R,
                self.policy.h_in: self.policy.prev_h})
            self.policy.prev_h = h_out
        return actions, values

    def reset(self):
        self.policy.reset()

    def save_policy(self):
        pass

    def train(self, ep_X, ep_A, ep_R, ep_adv, episode):
        # train network
        # TODO implement baseline
        # TODO TRPO optimizer
        # TODO tensorboard summaries
        # TODO GRU cells
        self.step_size*= 0.9995

        train_dict = {self.X: ep_X[1:], self.ADV: ep_adv[1:], self.A: ep_A[1:], self.R: ep_R[1:],
                      self.Aold: ep_A[:-1], self.Rold: ep_R[:-1]}
        if self.policy.recurrent:
            train_dict[self.policy.h_in] = self.policy.h_init
        pLoss, vLoss, Vfcn, ent, _ = self.session.run([self.pg_loss, self.vf_loss, self.policy.V, self.entropy, self.train_op],
			feed_dict=train_dict)

        if episode % 100 == 0:
            print('lr '+ str(self.step_size))

        pLoss_summary = tf.Summary(value=[tf.Summary.Value(tag='pLoss', simple_value=pLoss)])
        vLoss_summary = tf.Summary(value=[tf.Summary.Value(tag='vLoss', simple_value=vLoss)])
        reward_summary = tf.Summary(value=[tf.Summary.Value(tag='reward', simple_value=np.sum(ep_R))])
        self.summary_writer.add_summary(pLoss_summary, episode)
        self.summary_writer.add_summary(vLoss_summary, episode)
        self.summary_writer.add_summary(reward_summary, episode)
        self.summary_writer.flush()

        info = {}
        info['policy_loss'] = pLoss
        info['value_loss'] = vLoss
        info['policy_entropy'] = ent
        return info

    def observe_V(self, X, A, R):
        train_dict = {self.X: X, self.Aold: A, self.Rold: R}
        train_dict[self.policy.h_in] = self.policy.h_init
        V = self.session.run(self.policy.V, feed_dict=train_dict)

        return V
