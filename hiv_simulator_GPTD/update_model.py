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

    # to accumulate gradients
    gradBuffer = sess.run(QNet.tvars)  # get shapes of tensors
    for idx in range(len(gradBuffer)):
        gradBuffer[idx] *= 0


    # Gradient descent
    for e in range(batch_size):

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
                               feed_dict={QNet.state: state_valid,
                                          QNet.state_next: next_state_valid,
                                          QNet.nprec: noise_precision,
                                          QNet.is_online: False})

        # evaluate target model
        Qmax_target, phi_max_target = sess.run([Qtarget.Qmax, Qtarget.phi_max],
                                  feed_dict={Qtarget.state: state_valid,
                                             Qtarget.state_next: next_state_valid,
                                             Qtarget.amax_online: amax_online,
                                             Qtarget.nprec: noise_precision,
                                             QNet.is_online: False})

        # update model
        grads, loss, Qdiff = sess.run(
            [QNet.gradients, QNet.loss, QNet.Qdiff],
            feed_dict={QNet.context_state: state_train,
                       QNet.context_action: action_train,
                       QNet.context_reward: reward_train,
                       QNet.context_state_next: next_state_train,
                       QNet.state: state_valid,
                       QNet.action: action_valid,
                       QNet.reward: reward_valid,
                       QNet.state_next: next_state_valid,
                       QNet.done: done_valid,
                       QNet.amax_online: amax_online,
                       QNet.phi_max_target: phi_max_target,
                       QNet.Qmax_online: Qmax_target,
                       QNet.lr_placeholder: learning_rate,
                       QNet.nprec: noise_precision,
                       QNet.is_online: False})

        # fullbuffer.update(idxs[0], loss0/ len(valid))

        for idx, grad in enumerate(grads):  # grad[0] is gradient and grad[1] the variable itself
            gradBuffer[idx] += (grad[0] / batch_size)

        lossBuffer += loss

    # update summary
    feed_dict = dictionary = dict(zip(QNet.gradient_holders, gradBuffer))
    feed_dict.update({QNet.lr_placeholder: learning_rate})

    # reduce summary size
    if episode % 10 == 0:
        # volume of cube encompassing trajectory
        state_coverage = np.max(state_sample, axis=0)- np.min(state_sample, axis=0)
        volume_coverage = np.prod(state_coverage)

        # update summary
        _, summaries_gradvar = sess.run([QNet.updateModel, QNet.summaries_gradvar], feed_dict=feed_dict)

        loss_summary = tf.Summary(value=[tf.Summary.Value(tag='Performance/Loss', simple_value=(lossBuffer / batch_size))])
        coverage_summary = tf.Summary(value=[tf.Summary.Value(tag='Exploration-Exploitation/State Coverage', simple_value=volume_coverage)])

        summary_writer.add_summary(loss_summary, episode)
        summary_writer.add_summary(coverage_summary, episode)
        summary_writer.add_summary(summaries_gradvar, episode)

        summary_writer.flush()
    else:
        _ = sess.run([QNet.updateModel], feed_dict=feed_dict)