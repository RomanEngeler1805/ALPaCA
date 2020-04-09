import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# read through the parameter files
start_file = 'tensorflow_22-29-30_03-20'
last_file = 'tensorflow_20-05-31_03-20'

directory = './logger/'

def return_data(filename, step):
    # read through the target distance files
    directory = './figures/'

    if os.path.isfile(directory+ filename +'/target_dist'+str(step)):
        target_distance = pd.read_csv(directory+ filename +'/target_dist'+str(step))
        arr = np.array([target_distance.values[:,0],
                       target_distance.values[:,1],
                       target_distance.values[:,2]])
        return np.transpose(arr)

    return []


if __name__ == "__main__":
    parameters = ['gamma', 'latent_space', 'hidden_space', 'rew_norm', 'noise_precision']

    df_list = ['file'] + parameters
    print(df_list)
    data_list = pd.DataFrame(columns=df_list)
    print(data_list)

    for filename in os.listdir(directory):
        # check if in file range
        if int(filename[-12:-10]) >= int(start_file[-8:-6]) and int(filename[-12:-10]) <= int(last_file[-8:-6]):
            # open file and loop over lines
            temp_list = pd.DataFrame({'file': filename[11:-4],
                                      parameters[0]: [0.],
                                      parameters[1]: [0],
                                      parameters[2]: [0],
                                      parameters[3]: [0.],
                                      parameters[4]: [0.]})
            data_list = data_list.append(temp_list)

            with open(directory + filename, "r") as file:
                for line in file:
                    # find substring by checking list
                    for param in parameters:
                        if line.find(param) >= 0:
                            pos = line.find('=') + 1
                            data_list.loc[data_list['file']==filename[11:-4], param] = float(line[pos:-1])

    print(data_list)
    print('-------')

    # plot mean reward of last 10k steps over gamma for step= 4
    sort_list = data_list.sort_values(by='rew_norm')

    plt.figure()
    for step in [4]:
        plot_array = np.zeros([data_list.shape[0], 2])

        for i in range(data_list.shape[0]):
            # read the data
            temp_data = return_data(sort_list.iloc[i]['file'], step=step)
            if len(temp_data) > 0:
                mean_dist = np.min(temp_data[:,1])#np.mean(temp_data[-5:, 1])
            else:
                mean_dist = 0

            if mean_dist > 0:
                plot_array[i, 0] = data_list.iloc[i]['rew_norm']
                plot_array[i, 1] = mean_dist

        plt.scatter(plot_array[:,0], plot_array[:, 1])

    #plt.xlim([0.85, 1.0])
    plt.xlim([1e-4, 1e-1])
    plt.xscale('log')
    plt.show()