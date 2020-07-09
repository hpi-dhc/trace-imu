# TRACE-IMU

SONAR (Service Oriented Nursing Activity Recignition) assists nurses by getting rid of the documentation burden and supports the nurse-patient interaction. SONAR uses acceleration data coming from the nurses physical activities.

## Installation Prerequisites

conda create --name name_of_your_choice python=3.7

conda activate name_of_your_choice

conda install -c anaconda pandas scikit-learn tensorflow-gpu=2.1.0

conda install -c conda-forge keras opencv

change line 506 of /path/to/env/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py from
_LOCAL_DEVICES = tf.config.experimental_list_devices() to
_LOCAL_DEVICES = [x.name for x in tf.config.list_logical_devices()]

pip install git+https://github.com/nghorbani/configer

pip install git+https://github.com/nghorbani/human_body_prior

### Starting Example

python train.py --dataset dip_imu --model ConvLSTM --data_path /home/Orhan.Konak/netstore/Data/ --gpu 1 --horizontal_flip

## Classification

The classification of the activities is accomplished by the following steps.

### Data Transformation

The Raw Acceleration Data is transformed to position data by integrating two times. Starting point is the origin. We get a data point in each timestemp in 3D:

![alt text][tracking]

[tracking]: https://github.com/KonakML/SONAR-ML-Acc-Image/blob/master/Pictures/tracking_imu.jpg "Tracking IMU 3D"

Transforming the time series into images allows to make usage of everything which already exists on imaages, like CNN. We can also make usage of GANs to generate more "data".

### Dimensionality Reduction

We can work on the 3D images or in order to explore the shape of the pattern, which is made by the movement of the nurse, we transform the 3D to 2D by using PCA:

![alt text][pca]

[pca]: https://github.com/KonakML/SONAR-ML-Acc-Image/blob/master/Pictures/PCA.png "PCA"

### Windowing

For further preparation we decrease the window size, which allows a faster activity detection, as well as reduced resources and energy needs. An interesting approach is also to convert the line segments into a density map. This helps to further decrease the image size (like pooling):

![alt text][heatmap]

[heatmap]: https://github.com/KonakML/SONAR-ML-Acc-Image/blob/master/Pictures/heatmap.png "Heatmap"

### Image Classification

Image classification is done by the deep learning library fastAI. We used a pretained resnet32 (or resnet50) as a basis (transfer learning) and just trained the last layer. Another alternative is to go for Neural Architecture Search (NAS), maybe ENAS.

### Sequencing

To make sure that it classifies the right activity we tried different overlapping window sizes (2, 4, 8, ... sec). The more it was concluded the same activity the higher the likelood of having the right activity. (HMM, CNN-LSTM)?

## Conclusion

Faster, more accurate, less parameters, less training time, ...? It has to be compared with state-of-the-art solutions.
