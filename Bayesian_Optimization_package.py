from __future__ import print_function
from __future__ import division

from sklearn.datasets import make_classification
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from scipy.stats import randint as sp_randint
from bayes_opt import BayesianOptimization
from sklearn.grid_search import GridSearchCV , RandomizedSearchCV
import sys
import numpy as np
import random
from sklearn.datasets import load_iris, load_digits


# Load data set and target values
data, target = make_classification(
    n_samples=1000,
    n_features=45,
    n_informative=12,
    n_redundant=7
)

#test_DF = load_iris()
test_DF = load_digits()

data = test_DF.data.astype(np.float64)
target = test_DF.target.astype(np.float64)

def svccv(C, gamma):
    val = cross_val_score(
        SVC(C=C, gamma=gamma, random_state=2),
        data, target, 'accuracy', cv=2
    ).mean()

    return val

def rfccv(n_estimators, min_samples_split, max_features):
    val = cross_val_score(
        RFC(n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=min(max_features, 0.999),
            random_state=2
        ),
        data, target, 'accuracy', cv=2
    ).mean()
    return val

if __name__ == "__main__":
    gp_params = {"alpha": 1e-5}

    svcBO = BayesianOptimization(svccv,{'C': (0.001, 100), 'gamma': (0.0001, 0.1)})
    svcBO.explore({'C': [0.001, 0.01, 0.1], 'gamma': [0.001, 0.01, 0.1]})

    #rfcBO = BayesianOptimization(rfccv,{'n_estimators': (10, 250),'min_samples_split': (2, 25),'max_features': (0.1, 0.999)})

    svcBO.maximize(n_iter=20, **gp_params)
    print('-' * 53)
    #rfcBO.maximize(n_iter=10, **gp_params)

    print('-' * 53)
    print('Final Results')
    print('SVC: %f' % svcBO.res['max']['max_val'])
    #print('RFC: %f' % rfcBO.res['max']['max_val'])

    print("===========================================================================================================")
    param_dist = {
        "C": sp_randint(1, 100),
        "gamma": np.arange(0.001, 0.1, 0.001)
    }
    n_iter_search = 100

    clf = SVC(random_state=2)

    clf = RandomizedSearchCV(clf, param_distributions=param_dist,
                         n_iter=n_iter_search, scoring='accuracy', cv=2)
    clf.fit(data, target)

    print(clf.grid_scores_)

    print("Best estimator found by grid search:")
    print(clf.best_estimator_)
    print(clf.grid_scores_)
    print(clf.best_score_)
    print(clf.best_params_)
    print(clf.scorer_)
