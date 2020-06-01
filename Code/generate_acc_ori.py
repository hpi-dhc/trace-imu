import numpy as np
import pandas as pd
import sys
from scipy.spatial.transform import Rotation
import torch
from tqdm import tqdm

name = sys.argv[1]

comp_device = torch.device("cpu")

amass_path = 'netstore/IIC/data/'
data = [np.load(amass_path + name + '_training.npz', allow_pickle=True), np.load(amass_path + name + '_validation.npz', allow_pickle=True)]
if name == 'dip_imu':
    data.append(np.load(amass_path + name + '_test.npz', allow_pickle=True))
data = {
    'file_id': np.concatenate([i['file_id'] for i in data]),
    'acceleration': np.concatenate([i['acceleration'] for i in data]),
    'orientation': np.concatenate([i['orientation'] for i in data])
}

dataset = {}
for key in tqdm(set(data['file_id'])):
    for i in range(len(data['acceleration'])):
        if key == data['file_id'][i]:
            current = np.concatenate((data['acceleration'][i], data['orientation'][i]), axis=1)
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
    dataset = relabel(amass_path + 'labels.csv')

np.savez(amass_path + name + '_acc_ori.npz', **dataset)

