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
import logging
import sys
import pandas as pd

def main():
    # TODO change to Flags arguments
    # TODO same parameters as in RL2 paper
    # TODO plot trajectories for inspection
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_eps', type=int, default=int(1.5e4), help='training episodes')
    parser.add_argument('--test_eps', type=int, default=1000, help='test episodes')
    parser.add_argument('--seed', type=int, default=1, help='experiment seed')

    # Training Hyperparameters
    parser.add_argument('--hidden', type=int, default=64, help='hidden layer dimensions')
    parser.add_argument('--gamma', type=float, default=0.90, help='discount factor')
    parser.add_argument('--action_space', type=int, default=3, help='action space')

    parser.add_argument('--vf_coef', type=float, default=0.5, help='hidden layer dimensions')
    parser.add_argument('--ent_coef', type=float, default=0.01, help='discount factor')
    parser.add_argument('--step_size', type=float, default=5e-5, help='action space')
    parser.add_argument('--lr_drop', type=float, default=0.9997, help='learning rate drop')

    args = parser.parse_args()

    # create file handler which logs even debug messages
    logger_dir = './logger/'
    if logger_dir:
        if not os.path.exists(logger_dir):
            os.makedirs(logger_dir)

    f = open(logger_dir+ time.strftime('%H-%M-%d_%m-%y'), 'w')
    args = parser.parse_args()
    for arg in vars(args):
        string = str(arg)+ ': '+ str(getattr(args, arg))+ '\n'
        f.write(string)
    f.close()

    trajectory_dir = 'figures/' + time.strftime('%H-%M-%d_%m-%y') + '/trajectory/'
    if not os.path.exists(trajectory_dir):
        os.makedirs(trajectory_dir)

    Vdir = 'figures/' + time.strftime('%H-%M-%d_%m-%y') + '/Value/'
    if not os.path.exists(Vdir):
        os.makedirs(Vdir)

    reward_dir = 'figures/' + time.strftime('%H-%M-%d_%m-%y') + '/'
    if not os.path.exists(reward_dir):
        os.makedirs(reward_dir)

    # environment
    env = environment(9)
    eval_env = environment(9)

    # network
    algo = A2C(env=env,
    	session=get_session(),
        policy_cls=LSTMPolicy,
        hidden_dim=args.hidden,
        action_dim=args.action_space,
        vf_coef = args.vf_coef,
        ent_coef = args.ent_coef,
        step_size = args.step_size,
        lr_drop = args.lr_drop,
    	scope='a2c')

    # analysis
    save_iter = args.train_eps // 20
    average_returns = []

    # report mean reward per episode
    reward_write_to_file = []

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

        # count reward
        reward_agent = []

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

            # count reward
            reward_agent.append(rew)

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

        reward_write_to_file.append(np.sum(np.asarray(reward_agent)))

        # preparing training inputs
        _, last_value = algo.get_actions(obs[None], np.array(action).reshape(-1), np.array(rew).reshape(-1))
        ep_X = np.asarray(ep_X, dtype=np.float32)
        ep_R = np.asarray(ep_R, dtype=np.float32)
        ep_A = np.asarray(ep_A, dtype=np.int32)
        ep_V = np.squeeze(np.asarray(ep_V, dtype=np.float32))
        ep_D = np.asarray(ep_D, dtype=np.float32)

        # policy gradient descent ====================================================
        if ep_D[-1] == 0:
            disc_rew = discount_with_dones(ep_R.tolist() + [np.squeeze(last_value)], ep_D.tolist() + [0], args.gamma)[:-1]
        else:
            disc_rew = discount_with_dones(ep_R.tolist(), ep_D.tolist(), args.gamma)
        ep_adv = disc_rew - ep_V

        # training
        algo.reset()
        _ = algo.train(ep_X=ep_X,
                       ep_A=ep_A,
                       ep_R=ep_R,
                       ep_adv=ep_adv,
                       episode=ep)
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
            _, Vright = algo.evaluate(X, A, R)

            A[1:] = 1 # going left
            algo.reset()
            _, Vleft = algo.evaluate(X[::-1], A, R)

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
    print('Mean Test Reward: {}'.format(np.mean(test_rewards[:])))
    print('Mean Test Reward: {}'.format(np.mean(np.array(test_rewards[:]))))

    df = pd.DataFrame(reward_write_to_file)
    df.to_csv(reward_dir + 'reward_per_episode', index=False)

if __name__=='__main__':
    main()