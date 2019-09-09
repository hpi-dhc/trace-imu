# SONAR-ML-Acc-Image

SONAR (Service Oriented Nursing Activity Recignition) assists nurses by getting rid of the documentation burden and supports the nurse-patient interaction. SONAR uses acceleration data coming from the nurses physical activities.

## Classification

The classification of the activities is accomplished by the following steps.

### Data Transformation

The Raw Acceleration Data is transformed to position data by integrating two times. Starting point is the origin. We get a data point in each timestemp in 3D.



### Dimensionality Reduction

PCA --> 2D Image

### Windowing

Small images through Windowing --> Heat map to reduce iamge size (like pooling)

### Image Classification

fastAI --> resnet32 or resnet50 transfer learning, just last layer train again

### Sequencing

Overlapping images --> HMM, LSTM
