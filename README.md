# learning-2-learn
L2L Sensorimotor Association problems (Nat. Neuro. 2023)

### Code dependencies
* Python (model)
  * Tensorflow
  * Numpy
  * Scipy
* Matlab (analysis)

### Model Training
Model can be trained via a call to the function train() in train.py

Call with default parameters used in the paper looks like:

```python
train(seed          = 0,
      batchSize     = 1,
      l2            = 0.0005,
      l2_wR          = 0.001,
      l2_wI          = 0.0001,
      l2_wO          = 0.1,
      learningRateInit = 0.0001,
      svBnd = 10.0,
      rType = runType.Full)
```
Here parameter(s):
* starting with $l2$ are regularization hyper-parameters
* $svBnd$ is the number of recurrent weight matrix singular values to regularize
* $learningRateInit$ is the learning rate
* $batchSize$ indicates 1 trial is learned at a time - keep this unchanged
* $rType$ indicates:
  * runType.Full : Train 1001 problems, one after the other
  * runType.ControlManifPert : Control condition for manifold perturbation; train 50 problems to uncover decision and stimulus subspaces; train on another 50 problems with frozen output weights.
  * runType.SSManifPert : S &rarr; S manifold perturbation; after uncovering decision and stimulus subspaces; train on another 50 problems with independent S &rarr; S perturbations
  * runType.DSManifPert : D &rarr; S manifold perturbation; after uncovering decision and stimulus subspaces; train on another 50 problems with independent D &rarr; S perturbations
  
Raw output files (trained weights - [file prefix *saved*] and trials to convergence - [file prefix *conv*]) are saved to the *data* folder. This folder is currently empty, but raw pre-trained model data in zipped format may be downloaded from *[here](https://drive.google.com/file/d/18aYDDRsktOVo5UU9bt14I0j0Ezba_0rw/view?usp=share_link)*.


### Model Analysis

Model analysis code that replicates main figures in paper are located in the *analysis* folder. This code saves processed data to the *results* folder.

Each matlab script in the *analysis* folder replicates 1 or 2 main figures. Each script that relies on time-consuming processing starts with the line:
```matlab
useSaved = true;
```
To re-run the processing on the raw files, set this flag to *false*. To rapidly create the figures from the saved processed data in the *results* folder, leave this flag set to *true*.
