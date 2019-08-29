import tensorflow as tf

class Qeval():
    def __init__(self, FLAGS, scope="QNetwork"):
        self.gamma = FLAGS.gamma
        self.action_dim = FLAGS.action_space
        self.state_dim = FLAGS.state_space
        self.hidden_dim = FLAGS.hidden_space
        self.latent_dim = FLAGS.latent_space
        self.cprec = FLAGS.prior_precision
        self.lr = FLAGS.learning_rate
        self.iter_amax = FLAGS.iter_amax
        self.scope = scope

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # build graph
            self._build_model()


    def _build_model(self):
        ''' constructing tensorflow model '''
        self.nprec = tf.placeholder(shape=[], dtype=tf.float32, name='noise_precision')

        # placeholders context data =======================================================
        self.context_phi = tf.placeholder(shape=[None, self.latent_dim, self.action_dim], dtype=tf.float32, name='context_phi')  # input
        self.context_action = tf.placeholder(shape=[None], dtype=tf.int32, name='action')
        self.context_reward = tf.placeholder(shape=[None], dtype=tf.float32, name='reward')

        (bsc, _, _) = tf.unstack(tf.to_int32(tf.shape(self.context_phi)))

        # placeholders predictive data ====================================================
        ## prediction data
        self.phi = tf.placeholder(shape=[None, self.latent_dim, self.action_dim], dtype = tf.float32, name='phi') # input
        self.action = tf.placeholder(shape=[None], dtype=tf.int32, name='action')
        self.reward = tf.placeholder(shape=[None], dtype=tf.float32, name='reward')

        (bs, _, _) = tf.unstack(tf.to_int32(tf.shape(self.phi)))

        # noise variance
        self.Sigma_e_context = 1. / self.nprec * tf.ones(bsc, name='noise_precision')
        self.Sigma_e = 1. / self.nprec * tf.ones(bs, name='noise_precision')

        # output layer (Bayesian) =========================================================
        # prior (updated via GD) ---------------------------------------------------------
        self.w0_bar = tf.get_variable('w0_bar', dtype=tf.float32, shape=[self.latent_dim, 1])
        self.L0_asym = tf.get_variable('L0_asym', dtype=tf.float32,
                                       initializer=tf.sqrt(self.cprec) * tf.ones(self.latent_dim))  # cholesky
        L0_asym = tf.linalg.diag(self.L0_asym)  # cholesky
        self.L0 = tf.matmul(L0_asym, tf.transpose(L0_asym))  # \Lambda_0

        self.wt = tf.get_variable('wt', shape=[self.latent_dim,1], trainable=False)
        self.Qout = tf.einsum('im,bia->ba', self.wt, self.phi, name='Qout')

        # posterior (analytical update) --------------------------------------------------
        context_taken_action = tf.one_hot(tf.reshape(self.context_action, [-1, 1]), self.action_dim, dtype=tf.float32)
        self.context_phi_taken = tf.reduce_sum(tf.multiply(self.context_phi, context_taken_action), axis=2)

        # update posterior if there is data
        self.wt_bar, self.Lt_inv = tf.cond(bsc > 0,
                                           lambda: self._update_posterior(self.context_phi_taken,
                                                                       self.context_reward),
                                           lambda: (self.w0_bar, tf.linalg.inv(self.L0)))

        self.sample_prior = self._sample_prior()
        self.sample_post = self._sample_posterior(tf.reshape(self.wt_bar, [-1, 1]), self.Lt_inv)

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
        # A = tf.linalg.cholesky(cov)
        V, U = tf.linalg.eigh(cov)
        z = tf.random_normal(shape=[self.latent_dim, 1])
        # x = mu + tf.matmul(A, z)
        x = mu + tf.matmul(tf.matmul(U, tf.sqrt(tf.linalg.diag(V))), z)
        return x

    def _update_posterior(self, phi_hat, reward):
        ''' update posterior distribution '''
        # I've done that differently than outlined in the write-up
        # since I don't like to put the noise variance inside the prior
        Le = tf.linalg.inv(tf.linalg.diag(self.Sigma_e_context))  # noise precision
        Lt = tf.matmul(tf.transpose(phi_hat), tf.matmul(Le, phi_hat)) + self.L0
        Lt_inv = tf.linalg.inv(Lt)  # posterior variance
        wt_unnormalized = tf.matmul(self.L0, self.w0_bar) + \
                          tf.matmul(tf.transpose(phi_hat), tf.matmul(Le, tf.reshape(reward, [-1, 1])))
        wt_bar = tf.matmul(Lt_inv, wt_unnormalized)  # posterior mean

        return wt_bar, Lt_inv