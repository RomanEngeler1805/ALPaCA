import numpy as np
import pandas as pd

# load
data = pd.read_csv('./figures/22-07-11_11-19/'+ 'regret_per_episode', sep=" ", header=None)

# analysis
arr = data.values
mean = np.mean(arr[-500:])
std = np.std(arr[-500:])

#
print('Regret: '+ str(mean)+ ' +- '+ str(std))
