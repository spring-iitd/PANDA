# PANDA
PANDA: **P**ractical **A**dversarial Attack Against I**n**trusion **D**etection **A**pplications

## Model Training
### Dataset
We're using the [UQ-IoT dataset](https://espace.library.uq.edu.au/view/UQ:17b44bb) for all the experiments. Why?
1. It was created using our in-house IoT testbed at The UQ, Australia.
2. A huge dataset containing diverse network attacks.
3. It will help replaying the adversarial attack on the platform.

### ML Model (Surrogate Model)
Our goal is to launch practical adversarial attack against Intrusion Detection Systems (IDS). To achieve this goal, we're creating an ML model pipeline that satisfies this two criteria:
1. The created ML pipeline (Packets -> Features -> Model) should give similar performance as the target IDS.
2. The created ML pipeline should facilitate easy gradient propagation to the input layer (problem space).

So that, this ML model can be attacked and the obtained adversarial attack traffic from the surrogate model can be repurposed to attack the target model, called transferability property of adversarial attacks.

Possible Surrogate Model:
1. CNN-based IDS
2. Temporal CNN
3. Other sequential models or Transformers

### Training
For training we used a small porton of the dataset. We used a small portion because the anomaly detection model is relatively small (for edge devices), and using a lot of data might result in overfitting.

### Testing
For testing, in the initial phase, we're using 120k benign samples and 6 attacks from different devices. Later, the entire dataset will be used for testing.

### Anomaly Score
Differs from model to model. Here, -ve of the reonstruction error.

### Threshold
This section covers, how we decide on the threshold for the anomaly detection. As a general convention, "An example is said to be anomalous, if the anomaly score is more than a pre-defined threshold". The definition of anomaly score is different for different models, e.g., for autoencoder, it's the reconstruction error. The threshold is calculated generally during training time with training set or validation set. Below are the various methods to calculate threshold.

#### Use a percentile of the reconstruction error distribution.
This is a simple and effective approach. We calculate the reconstruction error for each example in the training dataset and then identify the percentile of the reconstruction error distribution. For example, 95th percentile of the reconstruction error distribution. This means that any example with a reconstruction error greater than the 95th percentile would be considered anomalous.

## Adversarial Attack
So far, we have tried attacks such as:
- FGSM
- PGD

To run an attack, run the following command:

```python attack_main.py --device cpu --attack fgsm```

## Scope of Change in Code:
1. Addition of a new data representation method:\
    Change in files:
    ```datasets.py, preprocessing.py, models.py, train.py, attack_main.py```
2. Addition of a new model:\
    Change in files:
    ```models.py, train.py, attack_main.py```
3. Addition of a new attack:\
    Change in files:
    ```attacks.py, attack_main.py```