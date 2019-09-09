# SONAR-ML-Acc-Image

SONAR (Service Oriented Nursing Activity Recignition) assists nurses by getting rid of the documentation burden and supports the nurse-patient interaction. SONAR uses acceleration data coming from the nurses physical activities.

## Classification

The classification of the activities is accomplished by the following steps.

### Data Transformation

The Raw Acceleration Data is transformed to position data by integrating two times. Starting point is the origin. We get a data point in each timestemp in 3D:

![alt text][tracking]

[tracking]: https://github.com/KonakML/SONAR-ML-Acc-Image/blob/master/Pictures/tracking_imu.jpg "Tracking IMU 3D"

### Dimensionality Reduction

In order to explore the shape of the pattern, which is made by the movement of the nurse, we transform the 3D to 2D by using PCA:

![alt text][pca]

[pca]: https://github.com/KonakML/SONAR-ML-Acc-Image/blob/master/Pictures/PCA.png "Tracking IMU 3D"

### Windowing

Small images through Windowing --> Heat map to reduce iamge size (like pooling)

### Image Classification

fastAI --> resnet32 or resnet50 transfer learning, just last layer train again

### Sequencing

Overlapping images --> HMM, LSTM
