import tensorflow as tf

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
        with tf.variable_scope("latent", reuse=tf.AUTO_REUSE):
            # model architecture
            self.hidden1 = tf.contrib.layers.fully_connected(x, num_outputs=self.hidden_dim, activation_fn=None,
                                                        weights_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer),)
            hidden1 = self.activation(self.hidden1)

            self.hidden2 = tf.contrib.layers.fully_connected(hidden1, num_outputs=self.hidden_dim, activation_fn=None,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                        weights_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
            hidden2 = self.activation(self.hidden2)
            hidden2 = tf.concat([hidden2, tf.one_hot(a, self.action_dim, dtype=tf.float32)], axis=1)

            self.hidden3 = tf.contrib.layers.fully_connected(hidden2, num_outputs=self.hidden_dim, activation_fn=None,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                        weights_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
            hidden3 = self.activation(self.hidden3)

            # single head
            hidden5 = tf.contrib.layers.fully_connected(hidden3, num_outputs=self.latent_dim, activation_fn=None,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                        weights_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))

            # bring it into the right order of shape [batch_size, latent_dim, action_dim]
            hidden5_rs = tf.reshape(hidden5, [-1, self.action_dim, self.latent_dim])
            hidden5_rs = tf.transpose(hidden5_rs, [0, 2, 1])

        return hidden5_rs

    def state_trafo(self, state, action):
        ''' append action to the state '''
        state = tf.expand_dims(state, axis=1)
        state = tf.tile(state, [1, 1, self.action_dim])
        state = tf.reshape(state, [-1, self.state_dim])

        action = tf.one_hot(action, self.action_dim, dtype=tf.float32)

        state = tf.concat([state, action], axis = 1)

        return state


    def _build_model(self):
        ''' constructing tensorflow model '''
        #
        self.lr_placeholder = tf.placeholder(shape=[], dtype=tf.float32, name='learning_rate')
        self.tau = tf.placeholder(shape=[], dtype=tf.float32, name='tau')
        self.nprec = tf.placeholder(shape=[], dtype=tf.float32, name='noise_precision')

        self.is_online = tf.placeholder_with_default(False, shape=[], name='is_online')
        self.use_cholesky = tf.placeholder_with_default(True, shape=[], name='use_cholesky')

        # placeholders context data =======================================================
        self.context_state = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name='state')  # input
        self.context_action = tf.placeholder(shape=[None], dtype=tf.int32, name='action')
        self.context_reward = tf.placeholder(shape=[None], dtype=tf.float32, name='reward')

        # append action s.t. it is an input to the network
        (bsc, _) = tf.unstack(tf.to_int32(tf.shape(self.context_state)))
        context_action_augm = tf.range(self.action_dim, dtype=tf.int32)
        context_action_augm = tf.tile(context_action_augm, [bsc])
        context_state = self.state_trafo(self.context_state, context_action_augm)

        # latent representation
        self.context_phi = self.model(context_state, context_action_augm)  # latent space

        # placeholders predictive data ====================================================
        ## prediction data
        self.state = tf.placeholder(shape=[None, self.state_dim], dtype = tf.float32, name='state') # input
        self.action = tf.placeholder(shape=[None], dtype=tf.int32, name='action')
        self.done = tf.placeholder(shape=[None], dtype=tf.float32, name='done')
        self.reward = tf.placeholder(shape=[None], dtype=tf.float32, name='reward')

        # append action s.t. it is an input to the network
        (bs, _) = tf.unstack(tf.to_int32(tf.shape(self.state)))
        action_augm = tf.range(self.action_dim, dtype=tf.int32)
        action_augm = tf.tile(action_augm, [bs])
        state = self.state_trafo(self.state, action_augm)

        # latent representation
        self.phi = self.model(state, action_augm) # latent space

        # noise variance
        self.Sigma_e_context = 1. / self.nprec * tf.ones(bsc, name='noise_precision')
        self.Sigma_e = 1. / self.nprec * tf.ones(bs, name='noise_precision')

        # output layer (Bayesian) =========================================================
        # prior (updated via GD) ---------------------------------------------------------
        self.w0_bar = tf.get_variable('w0_bar', dtype=tf.float32, shape=[self.latent_dim,1])
        self.L0_asym = tf.get_variable('L0_asym', dtype=tf.float32, initializer=tf.sqrt(self.cprec) * tf.ones(self.latent_dim))  # cholesky
        L0_asym = tf.linalg.diag(self.L0_asym)  # cholesky
        self.L0 = tf.matmul(L0_asym, tf.transpose(L0_asym))  # \Lambda_0

        self.wt = tf.get_variable('wt', shape=[self.latent_dim, 1], trainable=False)
        self.Qout = tf.einsum('jm,bjk->bk', self.wt, self.phi, name='Qout')

        # posterior (analytical update) --------------------------------------------------
        context_taken_action = tf.one_hot(tf.reshape(self.context_action, [-1, 1]), self.action_dim, dtype=tf.float32)
        self.context_phi_taken = tf.reduce_sum(tf.multiply(self.context_phi, context_taken_action), axis=2)

        taken_action = tf.one_hot(tf.reshape(self.action, [-1, 1]), self.action_dim, dtype=tf.float32)
        phi_taken = tf.reduce_sum(tf.multiply(self.phi, taken_action), axis=2)

        # update posterior if there is data
        self.wt_bar = tf.get_variable('wt_bar', initializer=self.w0_bar, trainable=False)
        self.Lt_inv = tf.get_variable('Lt_inv', initializer=tf.linalg.inv(self.L0), trainable=False) # covariance
        self.wt_unnorm = tf.get_variable('wt_unnorm', initializer=tf.matmul(self.L0, self.w0_bar), trainable=False)

        wt_bar, Lt_inv = tf.cond(bsc > 0,
                                 lambda: tf.cond(self.is_online,
                                                 lambda: self._update_posterior_online(self.context_phi_taken, self.context_reward),
                                                 lambda: self._update_posterior(self.context_phi_taken, self.context_reward)),
                                 lambda: (self.w0_bar, tf.linalg.inv(self.L0)))

        self.w_assign = tf.assign(self.wt_bar, wt_bar)
        self.L_assign = tf.assign(self.Lt_inv, Lt_inv)

        self.sample_prior = self._sample_prior()
        with tf.control_dependencies([self.w_assign, self.L_assign]):
            self.sample_post = self._sample_posterior(tf.reshape(self.wt_bar, [-1, 1]), self.Lt_inv)
        self.reset_post = self._reset_posterior()

        # loss function ==================================================================
        # current state -------------------------------------
        self.Q = tf.einsum('im,bi->b', wt_bar, phi_taken, name='Q')

        # next state ----------------------------------------
        self.Qtarget = self.reward

        # Q(s',a*)+ r- Q(s,a)
        self.Qdiff = self.Qtarget - self.Q

        # predictive covariance
        self.phi_hat = phi_taken

        Sigma_pred = tf.einsum('bi,ij,bj->b', self.phi_hat, Lt_inv, self.phi_hat,
                               name='Sigma_pred') + self.Sigma_e  # column vector
        logdet_Sigma = tf.reduce_sum(tf.log(Sigma_pred))

        # loss
        self.loss0 = tf.einsum('i,i->', self.Qdiff, self.Qdiff, name='loss0')
        self.loss1 = tf.einsum('i,ik,k->', self.Qdiff, tf.linalg.inv(tf.linalg.diag(Sigma_pred)), self.Qdiff, name='loss')
        self.loss2 = logdet_Sigma
        self.loss_reg = tf.losses.get_regularization_loss()

        self.loss = self.loss1+ self.loss2+ self.regularizer* self.loss_reg#+ 0.1* tf.reduce_sum(tf.square(tf.linalg.eigvalsh(Lt_inv)))

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_placeholder, beta1=0.9)

        self.tvars = tf.trainable_variables(scope=self.scope)  # [v for v in tf.trainable_variables() if v.name!='QNetwork/L0_asym:0']

        # gradient
        self.gradient_holders = []
        for idx, var in enumerate(self.tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.gradient_holders.append(placeholder)

        # symbolic gradient of loss w.r.t. tvars
        self.gradients = self.optimizer.compute_gradients(self.loss, self.tvars)
        #self.gradients = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gradients]

        #
        self.updateModel = self.optimizer.apply_gradients(zip(self.gradient_holders, self.tvars))

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
        for idx, var in zip(variables_names, self.gradient_holders):
            grad_hist_summary = tf.summary.histogram("/grad/hist/%s" % idx, var)
            grad_summaries.append(grad_hist_summary)

        # keep track of weights
        weight_summary = []
        for idx, var in zip(variables_names, self.tvars):
            weight_hist_summary = tf.summary.histogram("/weight/hist/%s" % idx, var)
            weight_summary.append(weight_hist_summary)

        # concat summaries
        self.summaries_gradvar = tf.summary.merge([grad_summaries, weight_summary])

    def _sample_prior(self):
        ''' sample wt from prior '''
        update_op = tf.assign(self.wt, self._sample_MN_cholesky(self.w0_bar, tf.matrix_inverse(self.L0)))
        return update_op

    def _sample_posterior(self, wt_bar, Lt_inv):
        ''' sample wt from posterior '''
        update_op = tf.assign(self.wt, tf.cond(self.use_cholesky,
                                               lambda: self._sample_MN_cholesky(wt_bar, Lt_inv),
                                               lambda: self._sample_MN_eigh(wt_bar, Lt_inv)))
        return update_op

    def _sample_MN_cholesky(self, mu, cov):
        ''' sample from multi-variate normal '''
        z = tf.random_normal(shape=[self.latent_dim, 1])
        A = tf.linalg.cholesky(cov + tf.diag(tf.constant(1e-6, shape=[self.latent_dim], dtype=tf.float32)))
        x = mu + tf.matmul(A, z)
        return x

    def _sample_MN_eigh(self, mu, cov):
        ''' sample from multi-variate normal '''
        z = tf.random_normal(shape=[self.latent_dim, 1])
        V, U = tf.linalg.eigh(cov)
        x = mu + tf.matmul(tf.matmul(U, tf.sqrt(tf.linalg.diag(V))), z)
        return x

    def _update_posterior(self, phi_hat, reward):
        ''' update posterior distribution '''
        # I've done that differently than outlined in the write-up
        # since I don't like to put the noise variance inside the prior
        Le = tf.linalg.inv(tf.linalg.diag(self.Sigma_e_context)) # noise precision
        Lt = tf.matmul(tf.transpose(phi_hat), tf.matmul(Le, phi_hat)) + self.L0
        Lt_inv = tf.linalg.inv(Lt) # posterior variance
        wt_unnormalized = tf.matmul(self.L0, self.w0_bar) + \
                          self.nprec*tf.matmul(tf.transpose(phi_hat), tf.reshape(reward, [-1, 1]))
        wt_bar = tf.matmul(Lt_inv, wt_unnormalized) # posterior mean

        return wt_bar, Lt_inv

    def _update_posterior_online(self, phi_hat, reward):
        # variance
        denum = 1./self.nprec + tf.einsum('bi,ij,bj->b', phi_hat, self.Lt_inv, phi_hat)
        denum = tf.reciprocal(denum)

        num = tf.einsum('ij,bj->bi', self.Lt_inv, phi_hat)
        num = tf.einsum('bi,bj->bij', num, num)

        Lt_inv = self.Lt_inv - tf.einsum('b,bij->ij', denum, num)

        # mean
        wt_unnorm = self.wt_unnorm + self.nprec * tf.reshape(tf.einsum('bi,b->i', phi_hat, reward), [-1, 1])  # XXXXXXXXXx
        update_op = tf.assign(self.wt_unnorm, wt_unnorm)

        with tf.control_dependencies([update_op]):
            wt_bar = tf.matmul(Lt_inv, wt_unnorm)

        return wt_bar, Lt_inv

    def _reset_posterior(self):
        update_op0 = tf.assign(self.wt_bar, self.w0_bar)
        update_op1 = tf.assign(self.Lt_inv, tf.linalg.inv(self.L0))
        update_op2 = tf.assign(self.wt_unnorm, tf.matmul(self.L0, self.w0_bar))

        update_op = tf.group([update_op0, update_op1, update_op2])

        return update_op


    def copy_params(self):
        # copy parameters
        update_op = []
        for (v_old, v_new) in zip(self.tvars, self.variable_holders):
            update_op.append(tf.assign(v_old, (1.- self.tau)* v_old+ self.tau* v_new))

        return update_op