import numpy as np
import pandas as pd

# load
data = pd.read_csv('figures/14-15-20_11-19/'+ 'reward_per_episode', sep=" ", header=None)
reward_agent = data.values
print(np.mean(reward_agent[-100:]))

#
N_tasks = 2
L_grid = 4 # 2*L+ 1
L_episode = 50
N_eval = 5000
rew_scaling = 1.2
bool_sparse = True

if bool_sparse == True:
	# sparse reward
	reward_max = 1./2.* N_tasks* rew_scaling* (L_episode+1- L_grid + L_episode+1- 3* L_grid)
	print('sparse max reward')
	print(reward_max)
else:
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

# analysis regret
mean_regret = reward_max- np.mean(reward_agent[-N_eval:])
stdv_regret = 1./np.sqrt(N_eval)* np.std(reward_agent[-N_eval:])

print('Regret: '+ str(mean_regret)+ ' +- '+ str(stdv_regret))

# analysis reward
mean_regret = np.mean(reward_agent[-N_eval:])/ reward_max
stdv_regret = 1./np.sqrt(N_eval)* np.std(reward_agent[-N_eval:])/ reward_max

print('Normalized reward: '+ str(mean_regret)+ ' +- '+ str(stdv_regret))
