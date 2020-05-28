import pickle as pkl
import os
import argparse
from models.cnn import CNN
from models.lstm import LSTM
from models.conv_lstm import ConvLSTM
from data import ImageDataset, TimeSeriesDataset, ImageTimeSeriesDataset

GPU = 2
if GPU is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

PARAMETERS = pkl.load(open('/home/Pit.Wegner/netstore/IIC/data/params.pkl', 'rb'))

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, help="the dataset to train on",
                choices=['biomotion', 'dip_imu'])
ap.add_argument("--model", required=True, help="the model to be used for training",
                choices=['CNN', 'LSTM', 'ConvLSTM'])
ap.add_argument("--horizontal_flip", help="data augmentation option for CNN training", action='store_true')
ap.add_argument("--vertical_flip", help="data augmentation option for CNN training", action='store_true')
ap.add_argument("--rotation_range", type=int, help="data augmentation option for CNN training")
ap.add_argument("--discrete", type=bool, help="whether to generate binary or continuous heatmap pixels in CNN training")
ap.add_argument("--modalities", help="comma-separated list of modalities to use for training")
ap.add_argument("--data_type", help="type of data input for LSTM training",
                choices=["raw", "trajectory"])

args = vars(ap.parse_args())

modalities = None
if args['modalities'] is not None:
    modalities = args['modalities'].split(',')

augmentation_options = {}
if args['rotation_range'] is not None:
    augmentation_options['rotation_range'] = args['rotation_range']
if args['horizontal_flip']:
    augmentation_options['horizontal_flip'] = True
if args['vertical_flip']:
    augmentation_options['vertical_flip'] = True

if args['model'] == 'CNN':
    dataset = ImageDataset(args['dataset'], PARAMETERS, modalities)
    folds, X, y = dataset.get_dataset()
    model = CNN('1', y.shape[1], X.shape[1:], augmentation=augmentation_options)
    for i, (train, val) in enumerate(folds):
        print("Training Fold", i)
        model.create_model()
        model.fit(X[train], y[train], X[val], y[val])

elif args['model'] == 'LSTM':
    if args['data_type'] is not None:
        dataset = TimeSeriesDataset(args['dataset'], PARAMETERS, modalities=modalities, kind=args['data_type'])
    else:
        dataset = TimeSeriesDataset(args['dataset'], PARAMETERS, modalities=modalities)
    folds, X, y = dataset.get_dataset()
    model = LSTM('1', y.shape[1], X.shape[1:])
    for i, (train, val) in enumerate(folds):
        print("Training Fold", i)
        model.create_model()
        model.fit(X[train], y[train], X[val], y[val])

elif args['model'] == 'ConvLSTM':
    dataset = ImageTimeSeriesDataset(args['dataset'], PARAMETERS, modalities=modalities)
    folds, X, y = dataset.get_dataset()
    print(X.shape, y.shape)
    model = ConvLSTM('1', y.shape[1], X.shape[1:])
    for i, (train, val) in enumerate(folds):
        print("Training Fold", i)
        model.create_model()
        model.fit(X[train], y[train], X[val], y[val])


