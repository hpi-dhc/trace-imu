import numpy as np
import sys
import pandas as pd
from tqdm import tqdm

name = sys.argv[1]

#amass_path = '/home/Pit.Wegner/netstore/IIC/data/'
amass_path = ''
data = np.load(amass_path + name + '_trajectories.npz', allow_pickle=True)
dataset = {}
for key in tqdm(set(data['labels'])):
    for i in range(len(data['positions'])):
        if key == data['labels'][i]:
            current = np.array(list(data['positions'][i])).swapaxes(0,1)
            if not key in dataset.keys():
                dataset[key] = current
            else:
                dataset[key] = np.concatenate((dataset[key], current))

def relabel(path='labels.csv'):
    to_export = {}
    t_labels = pd.read_csv(path)
    for label in dataset.keys():
        s = label.split("_", 3)
        if s[2] != 'b' and s[2] != 'c':
            k = "_".join(label.split("_", 2)[:2])
        else:
            k = "_".join(s[:3])

        timed = t_labels.loc[t_labels['sequence'] == k]
        for line in range(len(timed)):
            l0 = timed.iloc[::-1].iloc[line]
            key = label.split("_")[0] + '_' + l0['activity']
            if key in to_export.keys():
                key += '_d'
            if line == 0: 
                to_export[key] = dataset[label][l0['t']:]
            else:
                lf = timed.iloc[::-1].iloc[line-1]
                to_export[key] = dataset[label][l0['t']:lf['t']]
    return to_export

if name == 'dip_imu':
    dataset = relabel('/home/Pit.Wegner/netstore/IIC/data/' + 'labels.csv')

np.savez(name + '.npz', **dataset)
