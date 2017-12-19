import pandas as pd
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
import numpy as np
from tpot import TPOTClassifier, TPOTRegressor
import sys

#test_DF = load_iris()
test_DF = load_digits()

X_train, X_test, y_train, y_test = train_test_split(test_DF.data.astype(np.float64),
                                                    test_DF.target.astype(np.float64), train_size=0.75, test_size=0.25)

tpot_val = TPOTClassifier(generations=10, population_size=100, verbosity=2,n_jobs=4,scoring='accuracy')

tpot_val.fit(X_train, y_train)
print(tpot_val.score(X_test, y_test))
tpot_val.export('tpot_digits_pipeline.py')