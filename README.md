# PANDA
PANDA

## Dataset
We're using the [UQ-IoT dataset](https://espace.library.uq.edu.au/view/UQ:17b44bb) for all the experiments.
### Training
For training we used a small porton of the dataset. We used a small portion because the anomaly detection model is relatively small (for edge devices), and using a lot of data might result in overfitting.

### Testing
For testing, in the initial phase, we're using 120k benign samples and 6 attacks from different devices. Later, the entire dataset will be used for testing.

## Anomaly Score
Differs from model to model. Here, -ve of the reonstruction error.

## Threshold
This section covers, how we decide on the threshold for the anomaly detection. As a general convention, "An example is said to be anomalous, if the anomaly score is more than a pre-defined threshold". The definition of anomaly score is different for different models, e.g., for autoencoder, it's the reconstruction error. The threshold is calculated generally during training time with training set or validation set. Below are the various methods to calculate threshold.

### Use a percentile of the reconstruction error distribution.
This is a simple and effective approach. We calculate the reconstruction error for each example in the training dataset and then identify the percentile of the reconstruction error distribution. For example, 95th percentile of the reconstruction error distribution. This means that any example with a reconstruction error greater than the 95th percentile would be considered anomalous.

### Attack
So far, we have tried attacks such as:
- FGSM
- PGD

To run an attack, run the following command:

```python attack_main.py --device cpu --attack fgsm```