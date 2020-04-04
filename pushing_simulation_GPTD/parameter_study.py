import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# read through the parameter files
start_file = 'tensorflow_22-29-30_03-20'
last_file = 'tensorflow_20-05-31_03-20'

directory = './logger/'

parameters = ['gamma', 'latent_space', 'hidden_space', 'rew_norm', 'noise_precision']

df_list = ['file']+ parameters+ ['data']
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
                             parameters[4]: [0.],
                                    'data': [0.]})
        data_list = data_list.append(temp_list)

        with open(directory+ filename, "r") as file:
            for line in file:
                # find substring by checking list
                for param in parameters:
                    if line.find(param) >=0:
                        pos = line.find('=')+ 1
                        data_list.loc[data_list['file']==filename[11:-4], param] = float(line[pos:-1])

print(data_list)
print(len(data_list))

# read through the target distance files
directory = './figures/'
for filename in os.listdir(directory):
    # check if in file range
    if int(filename[6:8]) >= int(start_file[-8:-6]) and int(filename[6:8]) <= int(last_file[-8:-6]):
        #print(filename)

        # check if in file range
        for step in [4, 9, 14, 19, 24, 29, 34]:
            if os.path.isfile(directory+ filename +'/target_dist'+str(step)):
                target_distance = pd.read_csv(directory+ filename +'/target_dist'+str(step))
                print(np.asarray(target_distance.values[:, -1], dtype=float))
                data_list.loc[data_list['file']==filename, 'data'] = np.asarray(target_distance.values[:, -1], dtype=float)

                #colmn_name =


            else:
                print("File not exist")

print(data_list)
# loop over the time steps