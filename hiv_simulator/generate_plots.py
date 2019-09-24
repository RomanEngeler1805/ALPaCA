import numpy as np
import matplotlib.pyplot as plt

def generate_plots(save_dir, buffer, FLAGS, episode):
        trajectory_dir = save_dir + '/Trajectories/'
        histogram_dir = save_dir + '/Action_Histogram/'
        reward_dir = save_dir + '/Reward_Histogram/'
        states_dir = save_dir + '/State_Histogram/'

        trajectory_length = len(buffer.buffer)

        reward_train = np.zeros([trajectory_length, ])
        state_train = np.zeros([trajectory_length, FLAGS.state_space])
        next_state_train = np.zeros([trajectory_length, FLAGS.state_space])
        action_train = np.zeros([trajectory_length, ])
        done_train = np.zeros([trajectory_length, 1])

        # fill arrays
        for k, experience in enumerate(buffer.buffer):
            # [s, a, r, s', a*, d]
            state_train[k] = experience[0]
            action_train[k] = experience[1]
            reward_train[k] = experience[2]
            next_state_train[k] = experience[3]
            done_train[k] = experience[4]

        plt.figure()
        plt.plot(state_train)
        plt.xlabel('time')
        plt.ylabel('Concentration')
        plt.legend(['T1', 'T2', 'T1s', 'T2s', 'V', 'E'], loc='upper right')
        plt.savefig(trajectory_dir+ 'Episode_'+ str(episode))
        plt.close()

        plt.figure()
        plt.hist(action_train)
        plt.xlabel('action')
        plt.ylabel('count')
        plt.savefig(histogram_dir + 'Episode_' + str(episode))
        plt.close()

        plt.figure()
        plt.hist(reward_train)
        plt.xlabel('reward')
        plt.ylabel('count')
        plt.savefig(reward_dir + 'Episode_' + str(episode))
        plt.close()

        fig, ax = plt.subplots(ncols=5)
        for i in range(5):
            ax[i].hist(state_train[:,i])
            ax[i].set_xlabel('state '+str(i))
            ax[i].set_ylabel('count')
        plt.savefig(states_dir + 'Episode_' + str(episode))
        plt.close()