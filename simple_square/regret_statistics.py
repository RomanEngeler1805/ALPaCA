import numpy as np
import pandas as pd

# load
data = pd.read_csv('./figures/20-05-18_09-19/'+ 'reward_per_episode', sep=" ", header=None)
reward_agent = data.values
print(np.mean(reward_agent[-100:]))

#
N_tasks = 1
L_grid = 7 # 2*L+ 1
L_episode = 50
rew_scaling = 10

# sparse reward
reward_max = 1./2.* N_tasks* rew_scaling* (L_episode+1- L_grid + L_episode+1- 3* L_grid)
print('sparse max reward')
print(reward_max)

# dense reward
reward_max = 0.
# going correct direction
for i in range(1, L_grid+1):
	reward_max += 1.*(L_grid+ i)/(2* L_grid)
reward_max += (L_episode- L_grid)

# going wrong direction
reward_max += 1.*(L_grid- 1)/(2* L_grid)
for i in range(L_grid, 2* L_grid+ 1):
	reward_max += 1.*i/(2* L_grid)
reward_max += (L_episode- L_grid- 2)

# scaling
reward_max *= 1./2.* N_tasks* rew_scaling
print('dense max reward')
print(reward_max)

# analysis
regret = -(reward_agent - reward_max)
mean = np.mean(regret[-5000:])
std = np.std(regret[-5000:])

#
print('Regret: '+ str(mean)+ ' +- '+ str(std))
