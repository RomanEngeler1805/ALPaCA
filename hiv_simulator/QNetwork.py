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

    def model(self, x):
        ''' Embedding into latent space '''
        with tf.variable_scope("encoding", reuse=tf.AUTO_REUSE):
            # model architecture
            self.hidden1 = tf.contrib.layers.fully_connected(x, num_outputs=self.hidden_dim, activation_fn=None,
                                                        weights_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
            hidden1 = self.activation(self.hidden1)
            hidden1 = tf.contrib.layers.layer_norm(hidden1)

            self.hidden2 = tf.contrib.layers.fully_connected(hidden1, num_outputs=self.hidden_dim, activation_fn=None,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                        weights_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
            hidden2 = self.activation(self.hidden2)
            hidden2 = tf.contrib.layers.layer_norm(hidden2)

            self.hidden3 = tf.contrib.layers.fully_connected(hidden2, num_outputs=self.action_dim, activation_fn=None,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                        weights_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))

        return self.hidden3

    def _build_model(self):
        ''' constructing tensorflow model '''
        # hyperparameter
        self.lr_placeholder = tf.placeholder(shape=[], dtype=tf.float32, name='learning_rate')
        self.tau = tf.placeholder(shape=[], dtype=tf.float32, name='tau')
        self.nprec = tf.placeholder(shape=[], dtype=tf.float32, name='noise_precision')
        self.is_online = tf.placeholder_with_default(False, shape=[], name='is_online')

        # placeholders predictive data ====================================================
        ## prediction data
        self.state = tf.placeholder(shape=[None, self.state_dim], dtype = tf.float32, name='state') # input
        self.state_next = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name='next_state')  # input
        self.action = tf.placeholder(shape=[None], dtype=tf.int32, name='action')
        self.done = tf.placeholder(shape=[None], dtype=tf.float32, name='done')
        self.reward = tf.placeholder(shape=[None], dtype=tf.float32, name='reward')

        # append action s.t. it is an input to the network
        (bs, _) = tf.unstack(tf.to_int32(tf.shape(self.state)))

        # latent representation
        self.Q = self.model(self.state) # latent space
        self.Qnext = self.model(self.state_next)  # latent space

        # posterior (analytical update) --------------------------------------------------
        taken_action = tf.one_hot(self.action, self.action_dim, dtype=tf.float32)

        # loss function ==================================================================
        self.Qout = tf.identity(self.Q) # for only exploration

        self.Q = tf.reduce_sum(tf.multiply(self.Q, taken_action), axis=1)

        # Double Q learning
        self.max_action = tf.reshape(tf.argmax(self.Qnext, axis=1), [-1]) # max action from Q network
        self.amax_online = tf.placeholder(shape=[None], dtype=tf.int32, name='amax_online') # amax into Target network
        #
        self.Qmax = tf.reduce_sum(tf.multiply(self.Qnext, tf.one_hot(self.amax_online, self.action_dim)), axis=1) # Qmax from target network
        #
        self.Qmax_online = tf.placeholder(shape=[None], dtype=tf.float32, name='Qmax_target') # Qmax into Q network
        # Qtarget = r- Q(s,a)
        self.Qtarget = self.reward + self.gamma * tf.multiply(1 - self.done, self.Qmax_online)

        # Bellmann residual
        self.Qdiff = self.Qtarget - self.Q

        # loss
        self.loss = tf.einsum('b,b->', self.Qdiff, self.Qdiff, name='loss0')+ self.regularizer* tf.losses.get_regularization_loss(scope=self.scope)

        # optimizer =====================================================================
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


    def copy_params(self):
        # copy parameters
        update_op = []
        for (v_old, v_new) in zip(self.tvars, self.variable_holders):
            update_op.append(tf.assign(v_old, (1.- self.tau)* v_old+ self.tau* v_new))

        return update_op