import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#
a = np.load('./data/10-30-20_11-19/'+ 'episode_7200.npz')
#a = np.load('./data/17-44-20_11-19/'+ 'episode_5000.npz')
Value_plot = a['arr_0']
trajectory = a['arr_1']
target = a['arr_2']

#
fig, ax = plt.subplots(nrows=3, figsize=[8, 4])
idx1 = 0
idx2 = 4
idx3 = 12
'''
for i in range(10):
	im = ax[i].imshow(Value_plot[i].reshape(1,-1))
	im.set_clim(1.02* np.min(Value_plot[i]), 0.98*np.max(Value_plot[i]))
	print(np.round(Value_plot[i].reshape(1,-1), 3))
	ax[i].scatter(np.argmax(trajectory[i]), 0., s=50, color='b')
	ax[i].scatter(target, 0, s=10, color='r')
	ax[i].xaxis.set_ticklabels([''])
	ax[i].xaxis.set_visible(False)
	ax[i].yaxis.set_ticklabels([''])
	ax[i].yaxis.set_visible(False)
'''
im = ax[0].imshow(Value_plot[idx1].reshape(1,-1))
im.set_clim(1.02* np.min(Value_plot[idx1]), 0.98*np.max(Value_plot[idx1]))
ax[0].scatter(np.argmax(trajectory[idx1]), 0., s=500, color='b')
ax[0].scatter(target, 0, s=500, color='r')
ax[0].xaxis.set_ticklabels([''])
ax[0].xaxis.set_visible(False)
ax[0].yaxis.set_ticklabels([''])
ax[0].yaxis.set_visible(False)

im = ax[1].imshow(Value_plot[idx2].reshape(1,-1))
im.set_clim(1.02* np.min(Value_plot[idx2]), 0.98*np.max(Value_plot[idx2]))
ax[1].scatter(np.argmax(trajectory[idx2]), 0., s=500, color='b')
ax[1].scatter(target, 0, s=500, color='r')
ax[1].xaxis.set_ticklabels([''])
ax[1].xaxis.set_visible(False)
ax[1].yaxis.set_ticklabels([''])
ax[1].yaxis.set_visible(False)

im = ax[2].imshow(Value_plot[idx3].reshape(1,-1))
im.set_clim(1.02* np.min(Value_plot[idx3]), 0.98*np.max(Value_plot[idx3]))
ax[2].scatter(target, 0, s=500, color='r')
ax[2].scatter(np.argmax(trajectory[idx3]), 0., s=500, color='b')
ax[2].xaxis.set_ticklabels([''])
ax[2].xaxis.set_visible(False)
ax[2].yaxis.set_ticklabels([''])
ax[2].yaxis.set_visible(False)
plt.savefig('./sparse_value_thesis')
plt.show()
