from utils.parameters import parameters
from alpaca import alpaca
import numpy as np
import tensorflow as tf
import logging
import os
import time

if __name__ == "__main__":
    # load parameters
    FLAGS = parameters()

    log = logging.getLogger('Train')
    log.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create file handler which logs even debug messages
    logger_dir = './logger/'
    if logger_dir:
        if not os.path.exists(logger_dir):
            os.makedirs(logger_dir)

    fh = logging.FileHandler(logger_dir + 'tensorflow_' + time.strftime('%H-%M-%d_%m-%y') + '.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)

    # write hyperparameters to logger
    log.info('Parameters')
    for key in FLAGS.__flags.keys():
        log.info('{}={}'.format(key, getattr(FLAGS, key)))

    # random seed
    np.random.seed(FLAGS.random_seed)
    tf.set_random_seed(FLAGS.random_seed)
    alp = alpaca(FLAGS)
    print('run')
    alp.train()