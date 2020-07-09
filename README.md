# TRACE-IMU

TRACE-IMU (TRAjectory Classification Empowering IMU) transforms inertial sensor data that measure acceleration, orientation and angular velocity into movement trajectories and further into 2D heatmap images to take advantage of multiple image processing and classification techniques. To classify the images, a ConvLSTM network is used, which incorporates spatio-temporal correlations.

## Installation Prerequisites

```
conda create --name name_of_your_choice python=3.7
conda activate name_of_your_choice
conda install -c anaconda pandas scikit-learn tensorflow-gpu=2.1.0
conda install -c conda-forge keras opencv
```
change line 506 of /path/to/env/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py from
_LOCAL_DEVICES = tf.config.experimental_list_devices() to
```
_LOCAL_DEVICES = [x.name for x in tf.config.list_logical_devices()]
```
```
pip install git+https://github.com/nghorbani/configer
pip install git+https://github.com/nghorbani/human_body_prior
```

### Starting Example

```
python train.py --dataset dip_imu --model ConvLSTM --data_path /home/Orhan.Konak/netstore/Data/ --gpu 1 --horizontal_flip
```

## Classification

![alt text][pipeline]
[tracking]: https://github.com/KonakML/SONAR-ML-Acc-Image/blob/master/Pictures/tracking_imu.jpg "Tracking IMU 3D"
