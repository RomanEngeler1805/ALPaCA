import tensorflow as tf
import functools


class QNetwork():
    def __init__(self, FLAGS, scope="QNetwork"):
        self.gamma = FLAGS.gamma
        self.action_dim = FLAGS.action_space
        self.state_dim = FLAGS.state_space
        self.hidden_dim = FLAGS.hidden_space
        self.latent_dim = FLAGS.latent_space
        self.cprec = FLAGS.prior_precision
        self.lr = FLAGS.learning_rate
        self.regularizer = FLAGS.regularizer
        self.iter_amax = FLAGS.iter_amax
        self.grad_clip = FLAGS.grad_clip
        self.huber_d = FLAGS.huber_d
        self.scope = scope

        # activation function
        if FLAGS.non_linearity == 'sigm':
            self.activation = tf.nn.sigmoid
        elif FLAGS.non_linearity == 'tanh':
            self.activation = tf.nn.tanh
        elif FLAGS.non_linearity == 'elu':
            self.activation = tf.nn.elu
        elif FLAGS.non_linearity == 'leaky_relu':
            self.activation = tf.nn.leaky_relu
        elif FLAGS.non_linearity == 'selu':
            self.activation = tf.nn.selu
        else:
            self.activation = tf.nn.relu

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # build graph
            self._build_model()

    def model(self, x, a):
        ''' Embedding into latent space '''
        with tf.variable_scope("encoding", reuse=tf.AUTO_REUSE):
            # model architecture
            self.hidden1 = tf.contrib.layers.fully_connected(x, num_outputs=self.hidden_dim, activation_fn=None,
                                                        weights_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
            hidden1 = self.activation(self.hidden1)
            hidden1 = tf.contrib.layers.layer_norm(hidden1) # [batch, hidden]

            hidden1_ = tf.tile(tf.expand_dims(hidden1, axis=1), [1, 1, self.action_dim])
            hidden1_ = tf.reshape(hidden1_, [-1, self.hidden_dim])
            hidden1_ = tf.concat([hidden1_, tf.one_hot(a, self.action_dim, dtype=tf.float32)], axis=1)

            self.hidden2 = tf.contrib.layers.fully_connected(hidden1_, num_outputs=self.hidden_dim, activation_fn=None,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                        weights_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
            hidden2 = self.activation(self.hidden2)
            hidden2 = tf.contrib.layers.layer_norm(hidden2)

            self.hidden3 = tf.contrib.layers.fully_connected(hidden2, num_outputs=self.latent_dim, activation_fn=None,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                        weights_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))

            # bring it into the right order of shape [batch_size, latent_dim, action_dim]
            hidden3 = tf.reshape(self.hidden3, [-1, self.action_dim, self.latent_dim])
            hidden3 = tf.transpose(hidden3, [0, 2, 1])

        return hidden3

    def _build_model(self):
        ''' constructing tensorflow model '''
        # hyperparameter
        self.lr_placeholder = tf.placeholder(shape=[], dtype=tf.float32, name='learning_rate')
        self.tau = tf.placeholder(shape=[], dtype=tf.float32, name='tau')
        self.nprec = tf.placeholder(shape=[], dtype=tf.float32, name='noise_precision')
        self.is_online = tf.placeholder_with_default(False, shape=[], name='is_online')

        # placeholders predictive data ====================================================
        with tf.variable_scope("prediction", reuse=tf.AUTO_REUSE):
            # prediction data
            self.state = tf.placeholder(shape=[None, self.state_dim], dtype = tf.float32, name='state') # input
            self.state_next = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name='next_state')  # input
            self.action = tf.placeholder(shape=[None], dtype=tf.int32, name='action')
            self.reward = tf.placeholder(shape=[None], dtype=tf.float32, name='reward')
            self.exponent = tf.placeholder(shape=[None], dtype=tf.float32, name='exponent')
            self.done = tf.placeholder(shape=[None], dtype=tf.float32, name='done')

            # prepare action to concatenate after first hidden layer
            (bs, _) = tf.unstack(tf.to_int32(tf.shape(self.state)))
            action_augm = tf.range(self.action_dim, dtype=tf.int32)
            action_augm = tf.tile(action_augm, [bs])

            # noise variance
            self.Sigma_e = 1. / self.nprec * tf.ones(bs, name='noise_precision')

        ## latent representation
        self.phi = self.model(self.state, action_augm) # latent space
        self.phi_next = self.model(self.state_next, action_augm)  # latent space

        # placeholders context data =======================================================
        with tf.variable_scope("conditioning", reuse=tf.AUTO_REUSE):
            # conditioning data
            self.context_state = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name='state')  # input
            self.context_state_next = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name='next_state')  # input
            self.context_action = tf.placeholder(shape=[None], dtype=tf.int32, name='action')
            self.context_reward = tf.placeholder(shape=[None], dtype=tf.float32, name='reward')
            self.context_exponent = tf.placeholder(shape=[None], dtype=tf.float32, name='exponent')
            self.context_done = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='done')

            # append action s.t. it is an input to the network
            (bsc, _) = tf.unstack(tf.to_int32(tf.shape(self.context_state)))
            context_action_augm = tf.range(self.action_dim, dtype=tf.int32)
            context_action_augm = tf.tile(context_action_augm, [bsc])

            # noise variance
            self.Sigma_e_context = 1. / self.nprec * tf.ones(bsc, name='noise_precision')

        ## latent representation
        self.context_phi = self.model(self.context_state, context_action_augm)  # latent space
        self.context_phi_next = self.model(self.context_state_next, context_action_augm)  # latent space

        with tf.variable_scope("Bayesian", reuse=tf.AUTO_REUSE):
            # Bayesian layer
            self.w0_bar = tf.get_variable('w0_bar', dtype=tf.float32, shape=[self.latent_dim, 1])
            self.L0_asym = tf.get_variable('L0_asym', dtype=tf.float32, initializer=tf.sqrt(self.cprec) * tf.ones(self.latent_dim))  # cholesky
            L0_asym = tf.linalg.diag(self.L0_asym)  # cholesky
            self.L0 = tf.matmul(L0_asym, tf.transpose(L0_asym))  # \Lambda_0

            self.wt = tf.get_variable('wt', shape=[self.latent_dim, 1], trainable=False)
            self.Qout = tf.einsum('lm,bla->ba', self.wt, self.phi, name='Qout')  # exploration

            # posterior (analytical update) --------------------------------------------------
            context_taken_action = tf.one_hot(tf.reshape(self.context_action, [-1, 1]), self.action_dim, dtype=tf.float32)
            self.context_phi_taken = tf.reduce_sum(tf.multiply(self.context_phi, context_taken_action), axis=2)

            self.sample_prior = self._sample_prior()

            # posterior (analytical update) --------------------------------------------------
            taken_action = tf.one_hot(tf.reshape(self.action, [-1, 1]), self.action_dim, dtype=tf.float32)

            # loss function ==================================================================
            phi_taken = tf.reduce_sum(tf.multiply(self.phi, taken_action), axis=2)
            self.Q = tf.einsum('lm,bl->b', self.w0_bar, phi_taken, name='Qtaken')
            self.Qnext = tf.einsum('lm,bla->ba', self.w0_bar, self.phi_next, name='Qnext')

        with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
            # Double Q learning
            self.max_action = tf.one_hot(tf.reshape(tf.argmax(self.Qnext, axis=1), [-1, 1]), self.action_dim, dtype=tf.float32) # max action from Q network
            self.amax_online = tf.placeholder(shape=[None, 1, self.action_dim], dtype=tf.float32, name='amax_online')
            #
            self.phi_max = tf.reduce_sum(tf.multiply(self.phi_next, self.amax_online), axis=2)
            self.phi_max_target = tf.placeholder(shape=[None, self.latent_dim], dtype=tf.float32, name='phimax_target')
            #
            self.Qmax = tf.einsum('im,bi->b', self.w0_bar, self.phi_max)
            self.Qmax_online = tf.placeholder(shape=[None], dtype=tf.float32, name='Qmax_target') # Qmax into Q network

            # Qtarget = r- Q(s,a)
            self.Qtarget = self.reward + tf.pow(self.gamma, self.exponent) * tf.multiply(1 - self.done, self.Qmax_online)

            # Bellmann residual
            self.Qdiff = self.Qtarget - self.Q

            #
            self.phi_hat = phi_taken - self.gamma * self.phi_max_target

            Sigma_pred = tf.einsum('bi,ij,bj->b', self.phi_hat, tf.linalg.inv(self.L0), self.phi_hat, name='Sigma_pred') + self.Sigma_e  # column vector
            logdet_Sigma = tf.reduce_sum(tf.log(Sigma_pred))

            # loss
            self.loss1 = tf.einsum('i,ik,k->', self.Qdiff, tf.linalg.inv(tf.linalg.diag(Sigma_pred)), self.Qdiff, name='loss1')
            self.loss2 = logdet_Sigma

            # tf.losses.huber_loss(labels, predictions, delta=100.)
            self.loss = self.loss1 + self.loss2+ self.regularizer* tf.losses.get_regularization_loss(scope=self.scope)

        # optimizer =====================================================================
        with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_placeholder, beta1=0.9)

            self.tvars = tf.trainable_variables(scope=self.scope)  # [v for v in tf.trainable_variables() if v.name!='QNetwork/L0_asym:0']

            # gradient
            self.gradient_holders = []
            for idx, var in enumerate(self.tvars):
                placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
                self.gradient_holders.append(placeholder)

            # symbolic gradient of loss w.r.t. tvars
            gradients = self.optimizer.compute_gradients(self.loss, self.tvars)
            self.gradients = [(tf.clip_by_value(grad, -self.grad_clip, self.grad_clip), var) for grad, var in gradients]

            #
            self.updateModel = self.optimizer.apply_gradients(zip(self.gradient_holders, self.tvars))

        # copying weights =====================================================================
        # variable
        self.variable_holders = []
        for idx, var in enumerate(self.tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.variable_holders.append(placeholder)

        self.copyParams = self.copy_params()

        # summaries ========================================================================
        variables_names = [v.name for v in tf.trainable_variables()]

        # Keep track of gradient values
        grad_summaries = []
        grad_scalar_summaries = []
        for idx, var in zip(variables_names, self.gradient_holders):
            grad_hist_summary = tf.summary.histogram("/grad/hist/%s" % idx, var)
            grad_summaries.append(grad_hist_summary)

            grad_scalar_summary = tf.summary.scalar("Gradients/grad_%s" %idx, tf.reduce_sum(tf.sqrt(tf.square(var))))
            grad_scalar_summaries.append(grad_scalar_summary)

        # keep track of weights
        weight_summary = []
        for idx, var in zip(variables_names, self.tvars):
            weight_hist_summary = tf.summary.histogram("/weight/hist/%s" % idx, var)
            weight_summary.append(weight_hist_summary)

        # concat summaries
        self.summaries_gradvar = tf.summary.merge([grad_summaries, weight_summary, grad_scalar_summaries])


    def _sample_prior(self):
        ''' sample wt from prior '''
        update_op = tf.assign(self.wt, self._sample_MN(self.w0_bar, tf.matrix_inverse(self.L0)))
        return update_op

    def _sample_MN(self, mu, cov):
        ''' sample from multi-variate normal '''
        #A = tf.linalg.cholesky(cov)
        V, U = tf.linalg.eigh(cov)
        z = tf.random_normal(shape=[self.latent_dim,1])
        #x = mu + tf.matmul(A, z)
        x = mu+ tf.matmul(tf.matmul(U, tf.sqrt(tf.linalg.diag(V))), z)
        return x

    def copy_params(self):
        # copy parameters
        update_op = []
        for (v_old, v_new) in zip(self.tvars, self.variable_holders):
            update_op.append(tf.assign(v_old, (1.- self.tau)* v_old+ self.tau* v_new))

        return update_op