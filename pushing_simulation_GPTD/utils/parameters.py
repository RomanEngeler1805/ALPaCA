import tensorflow as tf
import sys

def parameters():
    # General Hyperparameters
    # general
    tf.flags.DEFINE_integer("batch_size", 2, "Batch size for training")
    tf.flags.DEFINE_float("gamma", 0.95, "Discount factor")
    tf.flags.DEFINE_integer("N_episodes", 60000, "Number of episodes")
    tf.flags.DEFINE_integer("N_tasks", 2, "Number of tasks")
    tf.flags.DEFINE_integer("L_episode", 50, "Length of episodes")

    # architecture
    tf.flags.DEFINE_integer("hidden_space", 128, "Dimensionality of hidden space")
    tf.flags.DEFINE_integer("latent_space", 4, "Dimensionality of latent space")
    tf.flags.DEFINE_string('non_linearity', 'leaky_relu', 'Non-linearity used in encoder')
    tf.flags.DEFINE_integer("nstep", 1, "n-step TD return")

    # domain
    tf.flags.DEFINE_integer("action_space", 7, "Dimensionality of action space")  # only x-y currently
    tf.flags.DEFINE_integer("state_space", 8, "Dimensionality of state space")  # [x,y,theta,vx,vy,vtheta]

    # posterior
    tf.flags.DEFINE_float("prior_precision", 0.1, "Prior precision (1/var)")
    tf.flags.DEFINE_float("noise_precision", 0.1, "Noise precision (1/var)")
    tf.flags.DEFINE_float("noise_precmax", 0.1, "Maximum noise precision (1/var)")
    tf.flags.DEFINE_integer("noise_Ndrop", 1, "Increase noise precision every N steps")
    tf.flags.DEFINE_float("noise_precstep", 1.0, "Step of noise precision s*=ds")

    tf.flags.DEFINE_integer("split_N", 20, "Increase split ratio every N steps")
    tf.flags.DEFINE_float("split_ratio", 0.6, "Initial split ratio for conditioning")
    tf.flags.DEFINE_float("split_ratio_max", 0.6, "Maximum split ratio for conditioning")
    tf.flags.DEFINE_integer("update_freq_post", 8, "Update frequency of posterior and sampling of new policy")

    # exploration
    tf.flags.DEFINE_float("eps_initial", 0., "Initial value for epsilon-greedy")
    tf.flags.DEFINE_float("eps_final", 0., "Final value for epsilon-greedy")
    tf.flags.DEFINE_float("eps_step", 0., "Multiplicative step for epsilon-greedy")

    # target
    tf.flags.DEFINE_float("tau", 0.008, "Update speed of target network")
    tf.flags.DEFINE_integer("update_freq_target", 1, "Update frequency of target network")

    # loss
    tf.flags.DEFINE_float("learning_rate", 1e-4, "Initial learning rate")
    tf.flags.DEFINE_float("lr_drop", 1.000, "Drop of learning rate per episode")
    tf.flags.DEFINE_float("lr_final", 1e-4, "Final learning rate")
    tf.flags.DEFINE_float("grad_clip", 1e3, "Absolute value to clip gradients")
    tf.flags.DEFINE_float("huber_d", 1e1, "Switch point from quadratic to linear")
    tf.flags.DEFINE_float("regularizer", 1e-4, "Regularization parameter") # X

    # reward
    tf.flags.DEFINE_float("rew_norm", 1e-1, "Normalization factor for reward")

    # memory
    tf.flags.DEFINE_integer("replay_memory_size", 5000, "Size of replay memory")
    tf.flags.DEFINE_integer("iter_amax", 1, "Number of iterations performed to determine amax")
    tf.flags.DEFINE_integer("save_frequency", 500, "Store images every N-th episode")

    #
    tf.flags.DEFINE_integer("random_seed", 2345, "Random seed for numpy and tensorflow")
    tf.flags.DEFINE_bool("load_model", False, "Load trained model")

    FLAGS = tf.flags.FLAGS
    FLAGS(sys.argv)

    return FLAGS