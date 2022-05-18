import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from ITMO_FS.filters.multivariate import MIM
from sklearn.preprocessing import KBinsDiscretizer

# error rate
def error_rate(xtrain, ytrain, x, opts):
    # parameters
    k     = opts['k']
    fold  = opts['fold']
    xt    = fold['xt']
    yt    = fold['yt']
    xv    = fold['xv']
    yv    = fold['yv']
    
    # Number of instances
    num_train = np.size(xt, 0)
    num_valid = np.size(xv, 0)
    # Define selected features
    xtrain  = xt[:, x == 1]
    ytrain  = yt.reshape(num_train)  # Solve bug
    xvalid  = xv[:, x == 1]
    yvalid  = yv.reshape(num_valid)  # Solve bug 
    acc = 0
    n = 0  
    
    # Training
    '''
    #1 KNN Classifier
    mdl     = KNeighborsClassifier(n_neighbors = k)
    mdl.fit(xtrain, ytrain)
    # Prediction
    ypred   = mdl.predict(xvalid)
    acc     += np.sum(yvalid == ypred) / num_valid
    n+=1
    '''
    
    #2 SVM Classifier
    sgdc = SGDClassifier(max_iter=1000, tol=0.01)
    sgdc.fit(xtrain, ytrain)
    ypred = sgdc.predict(xvalid)
    #print("SVM Accuracy: ", 100*(np.sum(yvalid == ypred) / num_valid))
    acc     += np.sum(yvalid == ypred) / num_valid
    n +=1
    
    
    #3 Naive Bayes
    gnb = GaussianNB().fit(xtrain, ytrain)
    ypred = gnb.predict(xvalid)
    #print("NB Accuracy: ", 100*(np.sum(yvalid == ypred) / num_valid))
    acc     += np.sum(yvalid == ypred) / num_valid
    n+=1
    
    acc/=n
    error   = 1 - acc
    return error


# Error rate & Feature size
def Fun(xtrain, ytrain, x, opts):
    # Parameters
    alpha    = 0.85
    beta  = 1-alpha
    beta     = 0.01 
    gamma     = 1 - alpha- beta
    selected_features = [i for i in range(len(x)) if x[i]==1]
    other_features = [i for i in range(len(x)) if i not in selected_features]
    #print(selected_features)
    est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy="uniform")
    
    # Original feature size
    max_feat = len(x)
    # Number of selected features
    num_feat = np.sum(x == 1)
    # Solve if no feature selected
    if num_feat == 0:
        cost  = 1
    else:
        # Get error rate
        error = error_rate(xtrain, ytrain, x, opts)
        # Objective function
        est.fit(xtrain)
        xtrain = est.transform(xtrain)
        x = MIM(np.array(selected_features), np.array(other_features), xtrain, ytrain)
        y = x.sum()/len(x)
        cost  = alpha * error + beta * (num_feat / max_feat) + gamma*(1-y)
        
    return cost
