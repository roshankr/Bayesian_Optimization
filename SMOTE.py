from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from sklearn.datasets import load_iris, load_digits
import numpy as np
import sys
import pandas as pd

test_DF = load_iris()
#test_DF = load_digits()

X = test_DF.data.astype(np.float64)
y = test_DF.target.astype(np.float64)

# X, y = make_classification(n_classes=2, class_sep=2,weights = [0.1, 0.9], n_informative = 3, n_redundant = 1,
#                            flip_y = 0,n_features = 20, n_clusters_per_class = 1, n_samples = 1000, random_state = 10)

print('Original dataset shape {}'.format(Counter(y)))
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_sample(X, y)
print('Resampled dataset shape {}'.format(Counter(y_res)))