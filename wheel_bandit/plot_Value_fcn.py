import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

def plot_Value_fcn(path, env_delta, sess, model, noise_precision, buffer=[]):
    # plot w* phi ---------------------------
    # generate grid (i.e. state) for plot
    env_rad = np.linspace(0., 1., 5)
    env_pha = np.linspace(0., 2. * np.pi, 20)
    mesh_rad, mesh_pha = np.meshgrid(env_rad, env_pha)
    env_state = np.concatenate([np.multiply(mesh_rad.reshape(-1, 1), np.cos(mesh_pha.reshape(-1, 1))),
                                np.multiply(mesh_rad.reshape(-1, 1), np.sin(mesh_pha.reshape(-1, 1)))], axis=1)

    # evaluate encoding and last layer
    wt_bar, phi = sess.run([model.w0_bar, model.phi], feed_dict={model.state: env_state, model.nprec: noise_precision})

    # check if observations available -> evaluate last layer
    if buffer:
        state_train = buffer[0]
        action_train = buffer[1]
        reward_train = buffer[2]
        next_state_train = buffer[3]
        done_train = buffer[4]

        # update
        wt_bar = sess.run([model.wt_bar])

    # dimension of state space
    ncols = phi.shape[2]

    # figure
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), ncols=ncols, figsize=[20, 5])

    for act in range(ncols):
        # plot value function
        Q_r = np.dot(phi[:, :, act], wt_bar).reshape(mesh_pha.shape)
        im = ax[act].contourf(mesh_pha, mesh_rad, Q_r)

        # plot exploration parameter
        ax[act].plot(env_pha, env_delta * np.ones([len(env_pha)]))

        # if observations available -> visualize data points
        if buffer:
            loc = np.where(action_train == act)  # to visualize which action network took
            rpos = np.linalg.norm(state_train[loc], axis=1)
            phipos = np.arctan2(state_train[loc, 1], state_train[loc, 0])
            ax[act].scatter(phipos, rpos, marker='o', color='r', s=1. + np.log(reward_train[loc]) * 10)

        # plot settings
        ax[act].set_xlim([0,1])
        ax[act].set_ylim([0,1])
        ax[act].set_xticks()
        ax[act].set_yticks()
        cb = fig.colorbar(im, ax=ax[act], orientation="horizontal", pad=0.1)
        tick_locator = ticker.MaxNLocator(nbins=4)
        cb.locator = tick_locator
        cb.update_ticks()

    plt.rc('font', size=16)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()