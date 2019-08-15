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

        # placeholders ====================================================================
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

        self.Sigma_e = 1. / self.nprec * tf.ones(bs, name='noise_precision')

        # output layer (Bayesian) =========================================================
        # prior (updated via GD) ---------------------------------------------------------
        self.w0_bar = tf.get_variable('w0_bar', dtype=tf.float32, shape=[self.latent_dim,1])
        self.Qout = tf.einsum('jm,bjk->bk', self.w0_bar, self.phi, name='Qout')

        # posterior (analytical update) --------------------------------------------------
        taken_action = tf.one_hot(tf.reshape(self.action, [-1, 1]), self.action_dim, dtype=tf.float32)
        phi_taken = tf.reduce_sum(tf.multiply(self.phi, taken_action), axis=2)

        # loss function ==================================================================
        # current state -------------------------------------
        self.Q = tf.einsum('im,bi->b', self.w0_bar, phi_taken, name='Q')

        # next state ----------------------------------------
        Qnext = tf.einsum('jm,bjk->bk', self.w0_bar, self.phi_next, name='Qnext')
        self.max_action = tf.one_hot(tf.reshape(tf.argmax(Qnext, axis=1), [-1, 1]), self.action_dim, dtype=tf.float32)

        self.amax_online = tf.placeholder(shape=[None, 1, self.action_dim], dtype=tf.float32, name='amax_online')
        self.Qmax = tf.einsum('im,bi->b', self.w0_bar, tf.reduce_sum(tf.multiply(self.phi_next, self.amax_online), axis=2))

        self.Qmax_target = tf.placeholder(shape=[None], dtype=tf.float32, name='Qmax_target')

        self.Qtarget = self.reward + self.gamma * tf.multiply(1 - self.done, self.Qmax_target)

        # Q(s',a*)+ r- Q(s,a)
        self.Qdiff = self.Qtarget - self.Q

        # loss
        self.loss0 = tf.einsum('i,i->', self.Qdiff, self.Qdiff, name='loss0')

        self.loss = self.loss0

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


    def copy_params(self):
        # copy parameters
        update_op = []
        for (v_old, v_new) in zip(self.tvars, self.variable_holders):
            update_op.append(tf.assign(v_old, (1.- self.tau)* v_old+ self.tau* v_new))

        return update_op