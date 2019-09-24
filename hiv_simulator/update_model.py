import numpy as np
import tensorflow as tf

def update_model(sess,
                 QNet,
                 Qtarget,
                 buffer,
                 summary_writer,
                 FLAGS,
                 episode,
                 batch_size=8,
                 split_ratio=0.5,
                 learning_rate=1e-4,
                 noise_precision=0.1):

    # to accumulate the losses across the batches
    lossBuffer = 0
    loss1Buffer = 0
    loss2Buffer = 0
    lossregBuffer = 0

    # to accumulate gradients
    gradBuffer = sess.run(QNet.tvars)  # get shapes of tensors
    for idx in range(len(gradBuffer)):
        gradBuffer[idx] *= 0


    # Gradient descent
    for e in range(batch_size):

        # probably not necessary: check
        sess.run(QNet.reset_post)

        # sample from larger buffer [s, a, r, s', d] with current experience not yet included
        experience = buffer.sample(1)

        L_episode = len(experience[0])

        state_sample = np.zeros((L_episode, FLAGS.state_space))
        action_sample = np.zeros((L_episode,))
        reward_sample = np.zeros((L_episode,))
        next_state_sample = np.zeros((L_episode, FLAGS.state_space))
        done_sample = np.zeros((L_episode,))

        # fill arrays
        for k, (s0, a, r, s1, d) in enumerate(experience[0]):
            state_sample[k] = s0
            action_sample[k] = a
            reward_sample[k] = r
            next_state_sample[k] = s1
            done_sample[k] = d

        # split into context and prediction set
        split = np.int(split_ratio * L_episode * np.random.rand())

        train = np.arange(0, split)
        valid = np.arange(split, L_episode)

        state_train = state_sample[train, :]
        action_train = action_sample[train]
        reward_train = reward_sample[train]
        next_state_train = next_state_sample[train, :]
        done_train = done_sample[train]

        state_valid = state_sample[valid, :]
        action_valid = action_sample[valid]
        reward_valid = reward_sample[valid]
        next_state_valid = next_state_sample[valid, :]
        done_valid = done_sample[valid]

        # TODO: this part is very inefficient due to many session calls and processing data multiple times
        # select amax from online network
        amax_online = sess.run(QNet.max_action,
                               feed_dict={QNet.context_state: state_train.reshape(-1, FLAGS.state_space),
                                          QNet.context_action: action_train.reshape(-1),
                                          QNet.context_reward: reward_train.reshape(-1),
                                          QNet.context_state_next: next_state_train.reshape(-1, FLAGS.state_space),
                                          QNet.context_done: done_train.reshape(-1, 1),
                                          QNet.state: state_valid,
                                          QNet.state_next: next_state_valid,
                                          QNet.nprec: noise_precision,
                                          QNet.is_online: False})

        # evaluate target model
        phi_max_target = sess.run(Qtarget.phi_max,
                                  feed_dict={Qtarget.context_state: state_train,
                                             Qtarget.context_action: action_train,
                                             Qtarget.context_reward: reward_train,
                                             Qtarget.context_state_next: next_state_train,
                                             Qtarget.state: state_valid,
                                             Qtarget.state_next: next_state_valid,
                                             Qtarget.amax_online: amax_online})

        # update model
        grads, loss, loss1, loss2, lossreg, Qdiff = sess.run(
            [QNet.gradients, QNet.loss, QNet.loss1, QNet.loss2, QNet.loss_reg, QNet.Qdiff],
            feed_dict={QNet.context_state: state_train.reshape(-1, FLAGS.state_space),
                       QNet.context_action: action_train.reshape(-1),
                       QNet.context_reward: reward_train.reshape(-1),
                       QNet.context_state_next: next_state_train.reshape(-1, FLAGS.state_space),
                       QNet.context_done: done_train.reshape(-1, 1),
                       QNet.state: state_valid,
                       QNet.action: action_valid,
                       QNet.reward: reward_valid,
                       QNet.state_next: next_state_valid,
                       QNet.done: done_valid,
                       QNet.phi_max_target: phi_max_target,
                       QNet.amax_online: amax_online,
                       QNet.lr_placeholder: learning_rate,
                       QNet.nprec: noise_precision,
                       QNet.is_online: False})

        # fullbuffer.update(idxs[0], loss0/ len(valid))

        for idx, grad in enumerate(grads):  # grad[0] is gradient and grad[1] the variable itself
            gradBuffer[idx] += (grad[0] / batch_size)

        lossBuffer += loss
        loss1Buffer += loss1
        loss2Buffer += loss2
        lossregBuffer += lossreg

    # update summary
    feed_dict = dictionary = dict(zip(QNet.gradient_holders, gradBuffer))
    feed_dict.update({QNet.lr_placeholder: learning_rate})

    # reduce summary size
    if episode % 10 == 0:
        # update summary
        _, summaries_gradvar = sess.run([QNet.updateModel, QNet.summaries_gradvar], feed_dict=feed_dict)

        loss_summary = tf.Summary(value=[tf.Summary.Value(tag='Loss', simple_value=(lossBuffer / batch_size))])
        loss1_summary = tf.Summary(value=[tf.Summary.Value(tag='Loss1', simple_value=(loss1Buffer / batch_size))])
        loss2_summary = tf.Summary(value=[tf.Summary.Value(tag='Loss2', simple_value=(loss2Buffer / batch_size))])
        lossreg_summary = tf.Summary(
            value=[tf.Summary.Value(tag='Loss reg', simple_value=(lossregBuffer / batch_size))])

        summary_writer.add_summary(loss_summary, episode)
        summary_writer.add_summary(loss1_summary, episode)
        summary_writer.add_summary(loss2_summary, episode)
        summary_writer.add_summary(lossreg_summary, episode)
        summary_writer.add_summary(summaries_gradvar, episode)

        summary_writer.flush()
    else:
        _ = sess.run([QNet.updateModel], feed_dict=feed_dict)

    # reset buffers
    for idx in range(len(gradBuffer)):
        gradBuffer[idx] *= 0

    lossBuffer *= 0.
    loss1Buffer *= 0.
    loss2Buffer *= 0.
    lossregBuffer *= 0.