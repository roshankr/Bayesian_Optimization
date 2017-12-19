from sklearn.datasets import make_classification
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import sklearn.gaussian_process as gp

data, target = make_classification(n_samples=2500,
                                   n_features=45,
                                   n_informative=15,
                                   n_redundant=5)

def sample_loss(params):
  C = params[0]
  gamma = params[1]

  # Sample C and gamma on the log-uniform scale
  model = SVC(C=10 ** C, gamma=10 ** gamma, random_state=12345)

  # Sample parameters on a log scale
  return cross_val_score(model=model,
                         X=data,
                         y=target,
                         scoring='roc_auc',
                         cv=3).mean()

print(np.shape(data))
print(target)


def bayesian_optimization(n_iters, sample_loss, xp, yp):
  """

  Arguments:
  ----------
    n_iters: int.
      Number of iterations to run the algorithm for.
    sample_loss: function.
      Loss function that takes an array of parameters.
    xp: array-like, shape = [n_samples, n_params].
      Array of previously evaluated hyperparameters.
    yp: array-like, shape = [n_samples, 1].
      Array of values of `sample_loss` for the hyperparameters
      in `xp`.
  """

  # Define the GP
  kernel = gp.kernels.Matern()
  model = gp.GaussianProcessRegressor(kernel=kernel,
                                      alpha=1e-4,
                                      n_restarts_optimizer=10,
                                      normalize_y=True)
  for i in range(n_iters):
    # Update our belief of the loss function
    model.fit(xp, yp)

    # sample_next_hyperparameter is a method that computes the arg
    # max of the acquisition function
    next_sample = sample_next_hyperparameter(model, yp)

    # Evaluate the loss for the new hyperparameters
    next_loss = sample_loss(next_sample)

    # Update xp and yp
