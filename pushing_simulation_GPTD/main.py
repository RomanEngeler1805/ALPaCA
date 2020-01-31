from utils.parameters import parameters
from alpaca import alpaca
import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    FLAGS = parameters()
    np.random.seed(FLAGS.random_seed)
    tf.set_random_seed(FLAGS.random_seed)
    alp = alpaca(FLAGS)
    print('run')
    alp.train()