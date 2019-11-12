import gym
import argparse
import numpy as np
import os
from utils import *
from policy import LSTMPolicy
from a2c import A2C
from bandit_environment import bandit_environment
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
    parser.add_argument('--train_eps', type=int, default=int(1e5), help='training episodes')
    parser.add_argument('--test_eps', type=int, default=500, help='test episodes')
    parser.add_argument('--seed', type=int, default=1, help='experiment seed')

    # Training Hyperparameters
    parser.add_argument('--hidden', type=int, default=64, help='hidden layer dimensions')
    parser.add_argument('--gamma', type=float, default=0., help='discount factor')
    parser.add_argument('--action_space', type=int, default=3, help='action space')

    parser.add_argument('--vf_coef', type=float, default=0.5, help='hidden layer dimensions')
    parser.add_argument('--ent_coef', type=float, default=0.01, help='discount factor')
    parser.add_argument('--step_size', type=float, default=5e-3, help='action space')

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

    V2dir = 'figures/' + time.strftime('%H-%M-%d_%m-%y') + '/Value2/'
    if not os.path.exists(V2dir):
        os.makedirs(V2dir)

    reward_dir = 'figures/' + time.strftime('%H-%M-%d_%m-%y') + '/'
    if not os.path.exists(reward_dir):
        os.makedirs(reward_dir)

    # environment
    env = bandit_environment(args.action_space)
    eval_env = bandit_environment(args.action_space)

    # network
    algo = A2C(env=env,
    	session=get_session(),
        policy_cls=LSTMPolicy,
        hidden_dim=args.hidden,
        action_dim=args.action_space,
        vf_ceof = args.vf_coef,
        ent_coef = args.ent_coef,
        step_size = args.step_size,
    	scope='a2c')

    # analysis
    save_iter = args.train_eps // 20
    average_returns = []

    # report mean reward per episode
    reward_write_to_file = []
    regret_write_to_file = []

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
        reward_opt = []
        reward_rand = []

        ep_X.append(0)
        ep_A.append(action)
        ep_V.append(value)
        ep_R.append(rew)
        ep_D.append(done)

        # online =====================================================================
        # inner loop
        while step < 40:
            action, value = algo.get_actions(obs[None], np.array(action).reshape(-1), np.array(rew).reshape(-1))
            new_obs, rew, done, rew_max, rew_rand = env._step(action)
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
            reward_opt.append(rew_max)
            reward_rand.append(rew_rand)

            if done == True:
                break

        #
        regret = (np.sum(np.asarray(reward_opt)) - np.sum(np.asarray(reward_agent))) / \
                 (np.sum(np.asarray(reward_opt)) - np.sum(np.asarray(reward_rand)))

        regret_write_to_file.append(regret)  # no need to divide by #tasks
        reward_write_to_file.append(np.sum(np.asarray(reward_agent)))

        # preparing training inputs
        _, last_value = algo.get_actions(obs[None], np.array(action).reshape(-1), np.array(rew).reshape(-1))
        ep_X = np.asarray(ep_X, dtype=np.float32)
        ep_R = np.asarray(ep_R, dtype=np.float32)
        ep_A = np.asarray(ep_A, dtype=np.int32)
        ep_V = np.squeeze(np.asarray(ep_V, dtype=np.float32))
        ep_D = np.asarray(ep_D, dtype=np.float32)

        # plot =======================================================================
        if ep % save_iter == 0 and ep != 0:
            # scalars
            print("Episode: {}".format(ep))
            print("MeanReward: {}".format(np.mean(average_returns[-50:])))

            # plot
            env_state = np.linspace(0, 1, 100)
            env_phase = env.phase
            env_psi = env._psi(env_state, env_phase)
            env_theta = env.theta

            fig, ax = plt.subplots(ncols=3)

            if args.action_space > 1:
                for act in range(args.action_space):
                    delta = np.where(ep_A == act)

                    env_r = env_theta[act] * env_psi[:, act]

                    ax[act].scatter(ep_X[delta], ep_V[delta], color='b')
                    ax[act].scatter(ep_X[delta], ep_R[delta], color='r')
                    ax[act].plot(env_state, env_r, color='r')
            else:
                env_r = env_theta * env_psi[:]

                ax[0].scatter(ep_X, ep_V, color='b')
                ax[0].scatter(ep_X, ep_R, color='r')
                ax[0].plot(env_state, env_r, color='r')

            plt.savefig(Vdir + 'episode_' + str(ep))
            plt.close()

            #
            V_increment = np.zeros(100)
            obs = np.linspace(0., 1., len(V_increment))

            for iter in range(len(V_increment)):
                _, V_increment[iter] = algo.evaluate(obs[iter].reshape(-1,1), np.array(action).reshape(-1), np.array(rew).reshape(-1))

            plt.figure()
            plt.plot(V_increment)
            plt.savefig(V2dir + 'episode_' + str(ep))
            plt.close()


        # policy gradient descent ====================================================
        if ep_D[-1] == 0:
            disc_rew = discount_with_dones(ep_R.tolist() + [np.squeeze(last_value)], ep_D.tolist() + [0], args.gamma)[:-1]
        else:
            disc_rew = discount_with_dones(ep_R.tolist(), ep_D.tolist(), args.gamma)
        ep_adv = disc_rew - ep_V

        # training
        algo.reset()
        _ = algo.train(ep_X=ep_X.reshape(-1,1),
                       ep_A=ep_A,
                       ep_R=ep_R,
                       ep_adv=ep_adv,
                       ep_Regret=regret,
                       episode=ep)
        average_returns.append(track_R)


        if ep % 5000 == 0:
            # evaluation =================================================================
            print('Evaluation ===================')
            # cumulative regret
            cumulative_regret = []

            # simple regret
            simple_regret = []


            for test_ep in range(args.test_eps):
                # new environment
                env._sample_env()
                obs = env._sample_state().copy()
                # reset hidden states
                algo.reset()
                # initialize
                step = 0
                action = 0
                rew = 0

                # inner loop (cumulative regret)
                reward_agent = 0
                reward_max = 0
                reward_rand = 0

                while step < 40:

                    action, value = algo.get_actions(obs[None], np.array(action).reshape(-1), np.array(rew).reshape(-1))
                    new_obs, rew, done, rew_max, rew_rand = eval_env._step(action)
                    obs = new_obs.copy()
                    step+= 1

                    # rewards
                    reward_agent += rew
                    reward_max += rew_max
                    reward_rand += rew_rand

                cumulative_regret.append((reward_max- reward_agent)/ (reward_max- reward_rand))

                # no updates to hidden state (simple regret)
                reward_agent = 0
                reward_max = 0
                reward_rand = 0

                for _ in range(40):
                    action, value = algo.evaluate(obs[None], np.array(action).reshape(-1), np.array(rew).reshape(-1))
                    new_obs, rew, done, rew_max, rew_rand = eval_env._step(action)
                    obs = new_obs.copy()

                    # rewards
                    reward_agent += rew
                    reward_max += rew_max
                    reward_rand += rew_rand

                simple_regret.append((reward_max- reward_agent)/ (reward_max- reward_rand))

            print('Mean Cumulative Regret: {}'.format(np.mean(np.asarray(cumulative_regret))))
            print('Mean Simple Regret: {}'.format(np.mean(np.asarray(simple_regret))))

            file = open(reward_dir + 'test_regret_per_episode', 'a')
            file.write('Episode'+ str(ep)+ ' =======================\n')
            file.write('Cumulative Regret\n')
            file.write('{:3.4f}% +- {:2.4f}%\n'.format(np.mean(np.asarray(cumulative_regret)), np.std(np.asarray(cumulative_regret))))
            file.write('Simple Regret\n')
            file.write('{:3.4f}% +- {:2.4f}%\n'.format(np.mean(np.asarray(simple_regret)), np.std(np.asarray(simple_regret))))
            file.close()

    df = pd.DataFrame(reward_write_to_file)
    df.to_csv(reward_dir + 'reward_per_episode', index=False)

    df = pd.DataFrame(regret_write_to_file)
    df.to_csv(reward_dir + 'regret_per_episode', index=False)

if __name__=='__main__':
    main()