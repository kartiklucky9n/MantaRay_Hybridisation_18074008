from ITMO_FS.filters.multivariate import generalizedCriteria
from sklearn.datasets import make_classification
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
dataset = make_classification(n_samples=100, n_features=20, n_informative=4, n_redundant=0, shuffle=False)

est = KBinsDiscretizer(n_bins=10, encode='ordinal')
data, target = np.array(dataset[0]), np.array(dataset[1])
print(data)
est.fit(data)
print("------------------------------------------")
print(data)
print(data.shape[0], data.shape[1])
data = est.transform(data)
selected_features = [1, 2, 3]
other_features = [i for i in range(0, data.shape[1]) if i not in selected_features]
x = generalizedCriteria(np.array(selected_features), np.array(other_features), data, target, 0.4, 0.3)
print(x.sum()/len(x))