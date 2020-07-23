# TRACE

TRACE-IMU (Trajectory Classification Employing IMU) is a combination of machine learning techniques in order to first generate poses in SMPL format from six IMU sensors using DIP, which are then tracked over time to form trajectories. These trajectories can be classified either as time series or images.

## Usage

For own data collection e.g. with GaitUp Physilog sensors, check the jupyter notebook `Preprocessing_Physilog.ipynb`.

## DIP

1. Clone `https://github.com/eth-ait/dip18` and cd into `train_and_eval`
2. Download pretrained model and DIP-IMU dataset from `http://dip.is.tuebingen.mpg.de/downloads` into `models` and `data` directories
3. Install dependencies
```
conda create --name name_of_your_choice_dip python=3.5
conda activate name_of_your_choice_dip
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0-cp35-cp35m-linux_x86_64.whl
conda install opencv
pip install numpy-quaternion
conda install numba
```
4. Generate poses from IMU dataset
```
python run_evaluation.py --system local --data_file own --model_id 1528208085 --save_dir ./models --eval_dir ./evaluation_results --datasets dip-imu --past_frames 20 --future_frames 5 --save_predictions
```

## Conversion to TRACE-compatible dataset

1. Download SMPL body models from `https://mano.is.tue.mpg.de/downloads`
2. Generate Raw Sensor Dataset
```
python generate_acc_ori.py dip_imu
```
3. Generate Trajectory Dataset
```
python generate_trajectories.py dip_imu
python transform_trajectories.py dip_imu
```

## TRACE

### Dependencies

```
conda create --name name_of_your_choice_trace python=3.7
conda activate name_of_your_choice_trace
conda install -c anaconda pandas scikit-learn tensorflow-gpu=2.1.0
conda install -c conda-forge keras opencv
pip install git+https://github.com/nghorbani/configer
pip install git+https://github.com/nghorbani/human_body_prior
```

change line 506 of `/path/to/env/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py` from
`_LOCAL_DEVICES = tf.config.experimental_list_devices()`
to
`_LOCAL_DEVICES = [x.name for x in tf.config.list_logical_devices()]`

### Start Training

The entrypoint for training is `train.py`. The parameters are explained executing `python train.py --help`. An example call looks like this:
```
python train.py --dataset dip_imu --model ConvLSTM --data_path /path/to/data/ --gpu 1 --horizontal_flip
```

## Classification

![Trajectory](/Images/trajectory.png)

![Trajectory](/Images/pipeline.png)
