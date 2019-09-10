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

[pca]: https://github.com/KonakML/SONAR-ML-Acc-Image/blob/master/Pictures/PCA.png "PCA"

### Windowing

For further preparation we decrease the window size, which allows a faster activity detection, as well as reduced resources and energy needs. An interesting approach is also to convert the line segments into a density map. This helps to further decrease the image size (like pooling):

![alt text][heatmap]

[heatmap]: https://github.com/KonakML/SONAR-ML-Acc-Image/blob/master/Pictures/heatmap.png "Heatmap"

### Image Classification

Image classification is done by the deep learning library fastAI. We used a pretained resnet32 (or resnet50) as a basis (transfer learning) and just trained the last layer. Another alternative is to go for Neural Architecture Search (NAS), maybe ENAS.

### Sequencing

To make sure that it classifies the right activity we tried different overlapping window sizes (2, 4, 8, ... sec). The more it was concluded the same activity the higher the likelood of having the right activity. (HMM, LSTM)?

## Conclusion

Faster, more accurate, less parameters, less training time, ...? It has to be compared with state-of-the-art solutions.
