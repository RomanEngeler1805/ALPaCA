import gym
import argparse
import numpy as np
import os
from utils import *
from policy import LSTMPolicy
from a2c import A2C
from environment import environment
import matplotlib.pyplot as plt
import time

def main():
    # TODO change to Flags arguments
    # TODO same parameters as in RL2 paper
    # TODO plot trajectories for inspection
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_eps', type=int, default=int(10e3), help='training episodes')
    parser.add_argument('--test_eps', type=int, default=300, help='test episodes')
    parser.add_argument('--seed', type=int, default=1, help='experiment seed')

    # Training Hyperparameters
    parser.add_argument('--hidden', type=int, default=256, help='hidden layer dimensions')
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor')
    args = parser.parse_args()

    trajectory_dir = 'figures/' + time.strftime('%H-%M-%d_%m-%y') + '/trajectory/'
    if not os.path.exists(trajectory_dir):
        os.makedirs(trajectory_dir)

    Vdir = 'figures/' + time.strftime('%H-%M-%d_%m-%y') + '/Value/'
    if not os.path.exists(Vdir):
        os.makedirs(Vdir)

    # environment
    env = environment(9)
    eval_env = environment(9)

    # network
    algo = A2C(env=env,
    	session=get_session(),
        policy_cls=LSTMPolicy,
        hidden_dim=args.hidden,
        action_dim=3,
    	scope='a2c')

    # analysis
    save_iter = args.train_eps // 20
    average_returns = []
    average_regret = []
    average_subopt = []

    #

    # episodes
    for ep in range(args.train_eps):
        env._sample_env()
        obs = env._sample_state().copy()
        done = False
        ep_X, ep_R, ep_A, ep_V, ep_D = [], [], [], [], []
        track_R = 0
        algo.reset()

        step = 0
        action = 0
        rew = 0
        value = 0

        ep_X.append(np.zeros([9]))
        ep_A.append(action)
        ep_V.append(value)
        ep_R.append(rew)
        ep_D.append(done)

        # inner loop
        while step < 50:
            action, value = algo.get_actions(obs[None], np.array(action).reshape(-1), np.array(rew).reshape(-1))
            new_obs, rew, done = env._step(action)
            track_R += rew

            ep_X.append(obs)
            ep_A.append(action)
            ep_V.append(value)
            ep_R.append(rew)
            ep_D.append(done)

            obs = new_obs.copy()

            step += 1
            if done == True:
                break

            if ep % save_iter == 0:
                plt.figure()
                plt.plot(np.argmax(np.asarray(ep_X)[1:], axis=1), 0.5 * np.ones(len(ep_X) - 1), 'b', marker='o', markersize=20)
                plt.plot(env.target, 0.5, 'r', marker='o', markersize=16)
                plt.xlim([-1, 9])
                plt.ylim([0, 1.0])
                plt.savefig(trajectory_dir + 'episode_' + str(ep)+ '_step_'+ str(step))
                plt.close()

        # preparing training inputs
        _, last_value = algo.get_actions(obs[None], np.array(action).reshape(-1), np.array(rew).reshape(-1))
        ep_X = np.asarray(ep_X, dtype=np.float32)
        ep_R = np.asarray(ep_R, dtype=np.float32)
        ep_A = np.asarray(ep_A, dtype=np.int32)
        ep_V = np.squeeze(np.asarray(ep_V, dtype=np.float32))
        ep_D = np.asarray(ep_D, dtype=np.float32)

        if ep_D[-1] == 0:
            disc_rew = discount_with_dones(ep_R.tolist() + [np.squeeze(last_value)], ep_D.tolist() + [0], args.gamma)[:-1]
        else:
            disc_rew = discount_with_dones(ep_R.tolist(), ep_D.tolist(), args.gamma)
        ep_adv = disc_rew - ep_V

        # training
        train_info = algo.train(ep_X=ep_X, ep_A=ep_A, ep_R=ep_R, ep_adv=ep_adv, episode=ep)
        average_returns.append(track_R)

        if ep % save_iter == 0 and ep != 0:
            print("Episode: {}".format(ep))
            print("MeanReward: {}".format(np.mean(average_returns[-50:])))

            X = np.eye(9)
            A = np.zeros([9])
            A[1:] = 2 # going right
            R = np.zeros([9])
            R[1:] = np.linspace(0.125, 1., 8) # going in the right direction
            algo.reset()
            Vright = algo.observe_V(X, A, R)

            A[1:] = 1 # going left
            algo.reset()
            Vleft = algo.observe_V(X[::-1], A, R)

            fig, ax = plt.subplots(nrows=2)
            ax[0].imshow(np.transpose(Vright))
            ax[1].imshow(np.transpose(Vleft))
            plt.savefig(Vdir + 'episode_' + str(ep))
            plt.close()


    # evaluation
    print('')
    test_regrets = []; test_rewards = []
    for test_ep in range(args.test_eps):
        env._sample_env()
        obs = env._sample_state().copy()

        algo.reset()
        done = False
        track_R = 0

        step = 0
        action = 0
        rew = 0

        # inner loop
        while step < 50:

            action, value = algo.get_actions(obs[None], np.array(action).reshape(-1), np.array(rew).reshape(-1))
            new_obs, rew, done = eval_env._step(action)
            obs = new_obs.copy()
            track_R += rew

            if done == True:
                break

            step+= 1

        test_rewards.append(track_R)
    print('Mean Test Cumulative Regret: {}'.format(np.mean(test_regrets)))
    print('Mean Test Reward: {}'.format(np.mean(test_rewards)))

if __name__=='__main__':
    main()