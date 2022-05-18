import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from MantaRayHybrid import jfs   # change this to switch algorithm 
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing


def normalize(df):
    x = df.copy()
    cols = df.columns.to_list()
    x = x.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    df.columns = cols
    return df
# load data
data  = pd.read_csv('datasets/preprocessed/parkinson_edit1.csv')

feat = data.iloc[:, :-1]
feat = normalize(feat)
feat  = feat.values
data = data.values
feat  = np.asarray(feat)
label = np.asarray(data[:, -1])

# split data into train & validation (70 -- 30)
xtrain, xtest, ytrain, ytest = train_test_split(feat, label, test_size=0.3, stratify=label)
fold = {'xt':xtrain, 'yt':ytrain, 'xv':xtest, 'yv':ytest}

# parameter
k    = 5     # k-value in KNN
N    = 10    # number of particles
T    = 100   # maximum number of iterations
opts = {'k':k, 'fold':fold, 'N':N, 'T':T}

start = time.time()
# perform feature selection
fmdl = jfs(feat, label, opts)
sf   = fmdl['sf']
end = time.time()
print("Time taken(in seconds): ", end - start)

# model with selected features
num_train = np.size(xtrain, 0)
num_valid = np.size(xtest, 0)
x_train   = xtrain[:, sf]
y_train   = ytrain.reshape(num_train)  # Solve bug
x_valid   = xtest[:, sf]
y_valid   = ytest.reshape(num_valid)  # Solve bug

#mdl       = KNeighborsClassifier(n_neighbors = k) 
acc = 0
n = 0 

#SDG
mdl = SGDClassifier(max_iter=1000, tol=0.01)
mdl.fit(x_train, y_train)
n+=1
y_pred    = mdl.predict(x_valid)
acc       += np.sum(y_valid == y_pred)  / num_valid

#NB
mdl = GaussianNB().fit(xtrain, ytrain)
mdl.fit(x_train, y_train)
n+=1
y_pred    = mdl.predict(x_valid)
acc       += np.sum(y_valid == y_pred)  / num_valid

# accuracy
acc       /= n
print("Mean Accuracy(SVM, NB):", 100 * acc)

# number of selected features
num_feat = fmdl['nf']
print("Feature Size:", num_feat)

# plot convergence
curve   = fmdl['c']
curve   = curve.reshape(np.size(curve,1))
x       = np.arange(0, opts['T'], 1.0) + 1.0

fig, ax = plt.subplots()
ax.plot(x, curve, 'o-')
ax.set_xlabel('Number of Iterations')
ax.set_ylabel('Fitness')
ax.set_title('Manta Ray')
ax.grid()
plt.show()

#mushroom dataset hit 22 genration to reach local minima