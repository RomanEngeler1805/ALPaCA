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
        self.iter_amax = FLAGS.iter_amax
        self.regularizer = FLAGS.regularizer
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
            #hidden2 = tf.concat([hidden2, tf.one_hot(a, self.action_dim, dtype=tf.float32)], axis=1)

            self.hidden3 = tf.contrib.layers.fully_connected(hidden2, num_outputs=self.hidden_dim, activation_fn=None,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                        weights_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
            hidden3 = self.activation(self.hidden3)

            # single head
            hidden5 = tf.contrib.layers.fully_connected(hidden3, num_outputs=self.latent_dim, activation_fn=None,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                        weights_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))

            # bring it into the right order of shape [batch_size, hidden_dim, action_dim]
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

        # for kl divergence to change learning dynamics
        self.w0_bar_old = tf.placeholder(tf.float32, shape=[self.latent_dim, 1], name='w0_bar_old')
        self.L0_asym_old = tf.placeholder(tf.float32, shape=[self.latent_dim], name='L0_asym_old')
        self.L0_old = tf.matmul(tf.diag(self.L0_asym_old), tf.diag(self.L0_asym_old))  # \Lambda_0

        # placeholders ====================================================================
        ## context data
        self.context_state = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name='state')  # input
        self.context_state_next = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name='next_state')  # input

        # append action s.t. it is an input to the network
        (bsc, _) = tf.unstack(tf.to_int32(tf.shape(self.context_state)))

        context_action_augm = tf.range(self.action_dim, dtype=tf.int32)
        context_action_augm = tf.tile(context_action_augm, [bsc])

        context_state = self.state_trafo(self.context_state, context_action_augm)
        context_state_next = self.state_trafo(self.context_state_next, context_action_augm)

        # latent representation
        self.context_phi = self.model(context_state, context_action_augm)  # latent space
        self.context_phi_next = self.model(context_state_next, context_action_augm)  # latent space

        self.context_action = tf.placeholder(shape=[None], dtype=tf.int32, name='action')
        self.context_done = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='done')
        self.context_reward = tf.placeholder(shape=[None], dtype=tf.float32, name='reward')

        ## prediction data
        self.state = tf.placeholder(shape=[None, self.state_dim], dtype = tf.float32, name='state') # input
        self.state_next = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name='next_state')  # input

        # append action s.t. it is an input to the network
        (bs, _) = tf.unstack(tf.to_int32(tf.shape(self.state)))

        action_augm = tf.range(self.action_dim, dtype=tf.int32)
        action_augm = tf.tile(action_augm, [bs])

        state = self.state_trafo(self.state, action_augm)
        state_next = self.state_trafo(self.state_next, action_augm)

        # latent representation
        self.phi = self.model(state, action_augm) # latent space
        self.phi_next = self.model(state_next, action_augm)  # latent space

        self.action = tf.placeholder(shape=[None], dtype=tf.int32, name='action')
        self.done = tf.placeholder(shape=[None], dtype=tf.float32, name='done')
        self.reward = tf.placeholder(shape=[None], dtype=tf.float32, name='reward')

        # noise variance
        self.nprec = tf.placeholder(shape=[], dtype=tf.float32, name='noise_precision')

        self.Sigma_e_context = 1. / self.nprec * tf.ones(bsc, name='noise_precision')
        self.Sigma_e = 1. / self.nprec * tf.ones(bs, name='noise_precision')

        # output layer (Bayesian) =========================================================
        self.wt = tf.get_variable('wt', shape=[self.latent_dim,1], trainable=False)
        self.Qout = tf.einsum('jm,bjk->bk', self.wt, self.phi, name='Qout')

        # prior (updated via GD) ---------------------------------------------------------
        self.w0_bar = tf.get_variable('w0_bar', dtype=tf.float32, shape=[self.latent_dim,1])
        self.L0_asym = tf.get_variable('L0_asym', dtype=tf.float32, initializer=tf.sqrt(self.cprec)*tf.ones(self.latent_dim)) # cholesky
        L0_asym = tf.linalg.diag(self.L0_asym)  # cholesky
        self.L0 = tf.matmul(L0_asym, tf.transpose(L0_asym))  # \Lambda_0

        self.sample_prior = self._sample_prior()

        # posterior (analytical update) --------------------------------------------------
        # phi(s, a)
        context_taken_action = tf.one_hot(tf.reshape(self.context_action, [-1, 1]), self.action_dim, dtype=tf.float32)
        self.context_phi_taken = tf.reduce_sum(tf.multiply(self.context_phi, context_taken_action), axis=2)

        taken_action = tf.one_hot(tf.reshape(self.action, [-1, 1]), self.action_dim, dtype=tf.float32)
        phi_taken = tf.reduce_sum(tf.multiply(self.phi, taken_action), axis=2)

        # update posterior if there is data
        self.wt_bar, self.Lt_inv = tf.cond(bsc > 0,
                                            lambda: self._max_posterior(self.context_phi_next, self.context_phi_taken,
                                                                        self.context_reward),
                                            lambda: (self.w0_bar, tf.linalg.inv(self.L0)))

        # sample posterior
        with tf.control_dependencies([self.wt_bar, self.Lt_inv]):
            self.sample_post = self._sample_posterior(tf.reshape(self.wt_bar, [-1, 1]), self.Lt_inv)

        # loss function ==================================================================
        # current state -------------------------------------
        self.Q = tf.einsum('im,bi->b', self.wt_bar, phi_taken, name='Q')

        # next state ----------------------------------------
        Qnext = tf.einsum('jm,bjk->bk', self.wt_bar, self.phi_next, name='Qnext')

        self.max_action = tf.one_hot(tf.reshape(tf.argmax(Qnext, axis=1), [-1, 1]), self.action_dim, dtype=tf.float32)

        # last factor to account for case that s is terminating state
        self.amax_online = tf.placeholder(shape=[None, 1, self.action_dim], dtype=tf.float32, name='amax_online')
        self.Qmax = tf.einsum('im,bi->b', self.wt_bar, tf.reduce_sum(tf.multiply(self.phi_next, self.amax_online), axis=2))


        self.Qmax_target = tf.placeholder(shape=[None], dtype=tf.float32, name='Qmax_target')

        self.Qtarget = self.reward + self.gamma * tf.multiply(1 - self.done, self.Qmax_target)
        #self.Qtarget = self.reward + self.gamma * tf.multiply(1- self.done, Qmax)
        #self.Qtarget = tf.stop_gradient(self.Qtarget)

        # Q(s',a*)+ r- Q(s,a)
        self.Qdiff = self.Qtarget - self.Q

        # phi_hat* Lt_inv* phi_hat --------------------------
        phi_max = tf.reduce_sum(tf.multiply(self.phi_next, self.amax_online), axis=2)
        phi_max = tf.stop_gradient(phi_max)

        self.phi_hat = phi_taken - self.gamma * phi_max
        #self.phi_hat = tf.stop_gradient(self.phi_hat)
        Sigma_pred = tf.einsum('bi,ij,bj->b', self.phi_hat, self.Lt_inv, self.phi_hat, name='Sigma_pred')+  self.Sigma_e # column vector
        logdet_Sigma = tf.reduce_sum(tf.log(Sigma_pred))

        # loss
        self.loss0 = tf.einsum('i,i->', self.Qdiff, self.Qdiff, name='loss0')
        self.loss1 = tf.einsum('i,ik,k->', self.Qdiff, tf.linalg.inv(tf.linalg.diag(Sigma_pred)), self.Qdiff, name='loss')
        self.loss2 = logdet_Sigma
        self.loss3 = -self.latent_dim- tf.linalg.logdet(self.L0)+ tf.linalg.logdet(self.L0_old)+\
                     tf.linalg.trace(tf.matmul(self.L0, tf.linalg.inv(self.L0_old)))+\
                     tf.matmul(tf.matmul(tf.linalg.transpose((self.w0_bar_old- self.w0_bar)), self.L0),(self.w0_bar_old- self.w0_bar))

        self.loss_reg = tf.losses.get_regularization_loss()#+ tf.nn.l2_loss(self.hidden5_W)+\
                        #tf.nn.l2_loss(self.w0_bar)+ tf.nn.l2_loss(self.L0)

        self.loss4 = tf.matmul(tf.reshape(self.L0_asym, [1,-1]), tf.reshape(self.L0_asym, [-1,1]))

        self.loss = self.loss1+ self.loss2 #+ FLAGS.regularizer* (self.loss_reg+ tf.nn.l2_loss(self.w0_bar))

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_placeholder)

        self.tvars = tf.trainable_variables(scope=self.scope)  # [v for v in tf.trainable_variables() if v.name!='QNetwork/L0_asym:0']


        # gradient
        self.gradient_holders = []
        for idx, var in enumerate(self.tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.gradient_holders.append(placeholder)

        # symbolic gradient of loss w.r.t. tvars
        self.gradients = self.optimizer.compute_gradients(self.loss, self.tvars)

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

        # prior last layer summaries
        hidden1_hist = tf.summary.histogram("hidden1", self.hidden1)
        hidden2_hist = tf.summary.histogram("hidden2", self.hidden2)
        hidden3_hist = tf.summary.histogram("hidden3", self.hidden3)

        # concat summaries
        self.summaries_gradvar = tf.summary.merge([grad_summaries, weight_summary])

        self.summaries_var = tf.summary.merge(weight_summary)

        self.summaries_encodinglayer = tf.summary.merge([hidden1_hist, hidden2_hist, hidden3_hist])


    def _sample_prior(self):
        ''' sample wt from prior '''
        update_op = tf.assign(self.wt, self._sample_MN(self.w0_bar, tf.matrix_inverse(self.L0)))
        return update_op

    def _sample_posterior(self, wt_bar, Lt_inv):
        ''' sample wt from posterior '''
        update_op = tf.assign(self.wt, self._sample_MN(wt_bar, Lt_inv))
        return update_op

    def _sample_MN(self, mu, cov):
        ''' sample from multi-variate normal '''
        #A = tf.linalg.cholesky(cov)
        V, U = tf.linalg.eigh(cov)
        z = tf.random_normal(shape=[self.latent_dim,1])
        #x = mu + tf.matmul(A, z)
        x = mu+ tf.matmul(tf.matmul(U, tf.sqrt(tf.linalg.diag(V))), z)
        return x

    def _update_posterior(self, phi_hat, reward):
        ''' update posterior distribution '''
        # I've done that differently than outlined in the write-up
        # since I don't like to put the noise variance inside the prior
        Le = tf.linalg.inv(tf.linalg.diag(self.Sigma_e_context)) # noise precision
        Lt = tf.matmul(tf.transpose(phi_hat), tf.matmul(Le, phi_hat)) + self.L0 # posterior precision
        Lt_inv = tf.linalg.inv(Lt) # posterior variance
        wt_unnormalized = tf.matmul(self.L0, self.w0_bar) + \
                          tf.matmul(tf.transpose(phi_hat), tf.matmul(Le, tf.reshape(reward, [-1, 1])))
        wt_bar = tf.matmul(Lt_inv, wt_unnormalized) # posterior mean

        return wt_bar, Lt_inv

    def _max_posterior(self, phi_next, phi_taken, reward):
        ''' determine wt_bar for calculating phi(s_{t+1}, a*) '''
        # determine phi(max_action) based on Q determined by sampling wt from prior
        _ = self._sample_prior() # sample wt
        Q_next = tf.einsum('ijk,jl->ik', phi_next, self.w0_bar)
        max_action = tf.one_hot(tf.reshape(tf.argmax(Q_next, axis=1), [-1, 1]), self.action_dim, dtype=tf.float32)
        phi_max = tf.reduce_sum(tf.multiply(phi_next, max_action), axis=2)

        # iterations
        for _ in range(self.iter_amax):
            # sample posterior
            phi_hat = phi_taken - self.gamma* phi_max

            # update posterior distributior
            wt_bar, Lt_inv = self._update_posterior(phi_hat, reward)

            # determine phi(max_action) based on Q determined by sampling wt from posterior
            _ = self._sample_posterior(wt_bar, Lt_inv)
            Q_next = tf.einsum('i,jik->jk', tf.reshape(self.wt, [-1]), phi_next) # XXXXX wt -> wt_bar

            max_action = tf.one_hot(tf.reshape(tf.argmax(Q_next, axis=1), [-1, 1]), self.action_dim, dtype=tf.float32)
            phi_max = tf.reduce_sum(tf.multiply(phi_next, max_action), axis=2)

        # stop gradient through context
        phi_max = tf.stop_gradient(phi_max)
        phi_hat = phi_taken - self.gamma * phi_max

        # update posterior distribution
        wt_bar, Lt_inv = self._update_posterior(phi_hat, reward)

        return wt_bar, Lt_inv



    def copy_params(self):
        # copy parameters
        update_op = []
        for (v_old, v_new) in zip(self.tvars, self.variable_holders):
            update_op.append(tf.assign(v_old, (1.- self.tau)* v_old+ self.tau* v_new))

        return update_op