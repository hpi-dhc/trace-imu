import pickle as pkl
import os
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import argparse
from models.cnn import CNN
from models.lstm import LSTM
from models.conv_lstm import ConvLSTM
from models.convlstm1d import ConvLSTM1D
from models.resnet_lstm import ResnetLSTM
from data import ImageDataset, TimeSeriesDataset, ImageTimeSeriesDataset
from keras import Model


ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, help="the dataset to train on",
                choices=['biomotion', 'dip_imu'])
ap.add_argument("--model", required=True, help="the model to be used for training",
                choices=['CNN', 'LSTM', 'ConvLSTM', 'ConvLSTM1D', 'ResnetLSTM'])
ap.add_argument("--horizontal_flip", help="data augmentation option for CNN training", action='store_true')
ap.add_argument("--vertical_flip", help="data augmentation option for CNN training", action='store_true')
ap.add_argument("--rotation_range", type=int, help="data augmentation option for CNN training")
#ap.add_argument("--discrete", help="whether to generate binary or continuous heatmap pixels in CNN training", action='store_true')
ap.add_argument("--modalities", help="comma-separated list of modalities to use for training")
ap.add_argument("--data_type", help="type of data input for LSTM training", choices=["raw", "trajectory"])
ap.add_argument("--subjects", help="subjects to be included for training")
ap.add_argument("--gpu", help="select GPU to train on, otherwise use CPU")
ap.add_argument("--data_path", required=True, help="select GPU to train on")

args = vars(ap.parse_args())

if args['gpu'] is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu']
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

PARAMETERS = pkl.load(open(args['data_path'] + 'params.pkl', 'rb'))

modalities = None
if args['modalities'] is not None:
    modalities = [int(i) for i in args['modalities'].split(',')]

subjects = None
if args['subjects'] is not None:
    subjects = [int(i) for i in args['subjects'].split(',')]

augmentation_options = {}
if args['rotation_range'] is not None:
    augmentation_options['rotation_range'] = args['rotation_range']
if args['horizontal_flip']:
    augmentation_options['horizontal_flip'] = True
if args['vertical_flip']:
    augmentation_options['vertical_flip'] = True

if args['model'] == 'CNN':
    dataset = ImageDataset(args['data_path'], args['dataset'], PARAMETERS, modalities)
    folds, X, y = dataset.get_dataset(include=subjects)
    model = CNN('1', y.shape[1], X.shape[1:], augmentation=augmentation_options)

elif args['model'] == 'ConvLSTM1D':
    if args['data_type'] is not None:
        dataset = TimeSeriesDataset(args['data_path'], args['dataset'], PARAMETERS, modalities=modalities, kind=args['data_type'])
    else:
        dataset = TimeSeriesDataset(args['data_path'], args['dataset'], PARAMETERS, modalities=modalities)
    folds, X, y = dataset.get_dataset(include=subjects)
    model = ConvLSTM1D('2', y.shape[1], X.shape[1:])

elif args['model'] == 'LSTM':
    if args['data_type'] is not None:
        dataset = TimeSeriesDataset(args['data_path'], args['dataset'], PARAMETERS, modalities=modalities, kind=args['data_type'])
    else:
        dataset = TimeSeriesDataset(args['data_path'], args['dataset'], PARAMETERS, modalities=modalities)
    folds, X, y = dataset.get_dataset(include=subjects)
    model = LSTM('2', y.shape[1], X.shape[1:])

elif args['model'] == 'ConvLSTM':
    dataset = ImageTimeSeriesDataset(args['data_path'], args['dataset'], PARAMETERS, modalities=modalities)
    folds, X, y = dataset.get_dataset(include=subjects)
    model = ConvLSTM('1', y.shape[1], X.shape[1:], augmentation=augmentation_options)

elif args['model'] == 'ResnetLSTM':
    dataset = ImageTimeSeriesDataset(args['data_path'], args['dataset'], PARAMETERS, modalities=modalities)
    folds, X, y = dataset.get_dataset(include=subjects)
    model = ResnetLSTM('1', y.shape[1], X.shape[1:])
print(X.shape)
accs = []
confusions = []
for i, (train, val) in enumerate(folds):
    print("Training Fold", i)
    model.create_model()
    model.fit(X[train], y[train], X[val], y[val])
    acc, conf = model.evaluate(X[val], y[val], dataset.enc)
    accs.append(acc)
    confusions.append(conf)
    print("Current average accuracy:", np.mean(accs))


print(dataset.enc)
print("Final Accuracy:", np.mean(accs))
for i in range(len(confusions)):
    print("Fold", i, "Accuracy:", accs[i])
    print(confusions[i])
