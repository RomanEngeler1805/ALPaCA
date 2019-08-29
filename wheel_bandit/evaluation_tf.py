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
        self.context_phi = tf.placeholder(shape=[None, self.latent_dim], dtype=tf.float32, name='state')  # input
        self.context_phi_next = tf.placeholder(shape=[None, self.latent_dim], dtype=tf.float32, name='next_state')  # input
        self.context_action = tf.placeholder(shape=[None], dtype=tf.int32, name='action')
        self.context_done = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='done')
        self.context_reward = tf.placeholder(shape=[None], dtype=tf.float32, name='reward')

        (bsc, _) = tf.unstack(tf.to_int32(tf.shape(self.context_state)))

        # placeholders predictive data ====================================================
        ## prediction data
        self.phi = tf.placeholder(shape=[None, self.state_dim], dtype = tf.float32, name='state') # input
        self.phi_next = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name='next_state')  # input
        self.action = tf.placeholder(shape=[None], dtype=tf.int32, name='action')
        self.done = tf.placeholder(shape=[None], dtype=tf.float32, name='done')
        self.reward = tf.placeholder(shape=[None], dtype=tf.float32, name='reward')

        (bs, _) = tf.unstack(tf.to_int32(tf.shape(self.state)))

        # noise variance
        self.Sigma_e_context = 1. / self.nprec * tf.ones(bsc, name='noise_precision')
        self.Sigma_e = 1. / self.nprec * tf.ones(bs, name='noise_precision')

        # output layer (Bayesian) =========================================================
        # prior (updated via GD) ---------------------------------------------------------
        self.w0_initial = tf.placeholder(shape=[self.latent_dim], dtype = tf.float32, name='w0_initial')
        self.L0_initial = tf.placeholder(shape=[self.latent_dim, self.latent_dim], dtype = tf.float32, name='w0_initial')

        self.w0_bar = tf.get_variable('w0_bar', dtype=tf.float32,
                                      initializer= tf.tile(tf.expand_dims(self.w0_initial, 0), [1,50]))
        self.L0 = tf.get_variable('L0', dtype=tf.float32,
                                      initializer=tf.tile(tf.expand_dims(self.L0_initial, 0), [1, 1, 50]))

        self.wt = tf.get_variable('wt', shape=[50, self.latent_dim], trainable=False)
        self.Qout = tf.einsum('bj,bj->b', self.wt, self.phi, name='Qout')

        # posterior (analytical update) --------------------------------------------------
        context_taken_action = tf.one_hot(tf.reshape(self.context_action, [-1, 1]), self.action_dim, dtype=tf.float32)
        self.context_phi_taken = tf.reduce_sum(tf.multiply(self.context_phi, context_taken_action), axis=2)

        taken_action = tf.one_hot(tf.reshape(self.action, [-1, 1]), self.action_dim, dtype=tf.float32)
        phi_taken = tf.reduce_sum(tf.multiply(self.phi, taken_action), axis=2)

        # update posterior if there is data
        self.wt_bar, self.Lt_inv = tf.cond(bsc > 0,
                                           lambda: self._update_posterior(self.context_phi_taken,
                                                                       self.context_reward),
                                           lambda: (self.w0_bar, tf.linalg.inv(self.L0)))

        self.sample_prior = self._sample_prior()
        self.sample_post = self._sample_posterior(tf.reshape(self.wt_bar, [-1, 1]), self.Lt_inv)


    def _sample_prior(self):
        ''' sample wt from prior '''
        w0 = tf.unstack(self.w0_bar)
        L0_inv = tf.unstack(tf.linalg.inv(self.L0))

        wt = []
        for (w, Linv) in zip(w0, L0_inv):
            wt.append(self._sample_MN(w, Linv))

        wt = tf.stack(wt)
        update_op = tf.assign(self.wt, wt)

        return update_op

    def _sample_posterior(self, wt_bar, Lt_inv):
        ''' sample wt from posterior '''
        w0 = tf.unstack(self.wt_bar)
        L0_inv = tf.unstack(Lt_inv)

        wt = []
        for (w, Linv) in zip(w0, L0_inv):
            wt.append(self._sample_MN(w, Linv))

        wt = tf.stack(wt)
        update_op = tf.assign(self.wt, wt)

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
        Le = 1./self.Sigma_e_context # noise precision

        Lt = Le* tf.einsum('bi,bj->bij', phi_hat, phi_hat)+ tf.tile(tf.expand_dims(self.L0, -1), axis=[1,1,50])

        Lt_inv = tf.linalg.inv(Lt) # posterior variance
        wt_unnormalized = tf.einsum('bij,jk->bi', self.L0, self.w0_bar) + \
                          Le* tf.einsum('bi,b->bi', phi_hat, reward)
        wt_bar = tf.einsum('bij,bj->bi',Lt_inv, wt_unnormalized)  # posterior mean

        return wt_bar, Lt_inv