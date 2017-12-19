from bayes_opt import BayesianOptimization
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec
#%matplotlib inline

def target(x):
    return np.exp(-(x - 2)**2) + np.exp(-(x - 6)**2/10) + 1/ (x**2 + 1)


KAPPA = 5
x = np.linspace(-2, 10, 1000)
y = target(x)

#plt.plot(x, y)

#A minimum number of 2 initial guesses is necessary to kick start the algorithms, these can either be random or user defined.

bo = BayesianOptimization(target, {'x': (-2, 10)})

#In this example we will use the Upper Confidence Bound (UCB) as our utility function. It has the free parameter κκ
#which control the balance between exploration and exploitation; we will set κ=5κ=5 which, in this case, makes the
#algorithm quite bold. Additionally we will use the cubic correlation in our Gaussian Process.

gp_params = {'corr': 'cubic'}
#bo.maximize(init_points=2, n_iter=0, acq='ucb', kappa=KAPPA, **gp_params)
bo.maximize(init_points=2, n_iter=0, acq='ucb', kappa=KAPPA)