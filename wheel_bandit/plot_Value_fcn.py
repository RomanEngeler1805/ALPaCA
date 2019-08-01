import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

def plot_Value_fcn(path, env, sess, model, noise_precision, buffer=[]):
    # plot w* phi
    env_rad = np.linspace(0., 1., 5)
    env_pha = np.linspace(0., 2. * np.pi, 20)
    mesh_rad, mesh_pha = np.meshgrid(env_rad, env_pha)
    env_state = np.concatenate([np.multiply(mesh_rad.reshape(-1, 1), np.cos(mesh_pha.reshape(-1, 1))),
                                np.multiply(mesh_rad.reshape(-1, 1), np.sin(mesh_pha.reshape(-1, 1)))], axis=1)

    env_delta = env.delta


    wt_bar, phi = sess.run([model.w0_bar, model.phi], feed_dict={model.state: env_state, model.nprec: noise_precision})

    if buffer:
        state_train = np.zeros([len(buffer), env.n_dim])
        action_train = np.zeros([len(buffer), ])
        reward_train = np.zeros([len(buffer), ])
        next_state_train = np.zeros((len(buffer), env.n_dim))
        done_train = np.zeros((len(buffer),1))

        # fill arrays
        for k, experience in enumerate(buffer):
            # [s, a, r, s', a*, d]
            state_train[k] = experience[0]
            action_train[k] = experience[1]
            reward_train[k] = experience[2]
            next_state_train[k] = experience[3]
            done_train[k] = experience[4]

        # update
        wt_bar = sess.run([model.wt_bar],
               feed_dict={model.context_state: state_train, model.context_action: action_train,
                          model.context_reward: reward_train, model.context_state_next: next_state_train,
                          model.context_done: done_train, model.nprec: noise_precision})

    ncols = phi.shape[2]

    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), ncols=ncols, figsize=[20, 5])

    for act in range(ncols):
        Q_r = np.dot(phi[:, :, act], wt_bar).reshape(mesh_pha.shape)
        im = ax[act].contourf(mesh_pha, mesh_rad, Q_r)
        ax[act].plot(env_pha, env_delta * np.ones([len(env_pha)]))

        if buffer:
            loc = np.where(action_train == act)  # to visualize which action network took
            rpos = np.linalg.norm(state_train[loc], axis=1)
            phipos = np.arctan2(state_train[loc, 1], state_train[loc, 0])
            ax[act].scatter(phipos, rpos, marker='o', color='r', s=1. + np.log(reward_train[loc]) * 10)

        cb = fig.colorbar(im, ax=ax[act], orientation="horizontal", pad=0.1)
        tick_locator = ticker.MaxNLocator(nbins=4)
        cb.locator = tick_locator
        cb.update_ticks()

    plt.rc('font', size=16)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()