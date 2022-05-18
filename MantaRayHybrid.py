#[2016]-"The whale optimization algorithm"]

import numpy as np
from numpy.random import rand
from functionHO import Fun


def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i,d] = lb[0,d] + (ub[0,d] - lb[0,d]) * rand()        
    
    return X


def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i,d] > thres:
                Xbin[i,d] = 1
            else:
                Xbin[i,d] = 0
    
    return Xbin


def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub
    
    return x


def jfs(xtrain, ytrain, opts):
    # Parameters
    ub    = 1
    lb    = 0
    thres = 0.5
    #b     = 1       # constant
    somersault_range = 2
    N        = opts['N']
    max_iter = opts['T']
    if 'b' in opts:
        b    = opts['b']
    alpha = 5

    dyn_feedback = 0
    n_changes = int(N/2)
    feedback_max=10
    
    # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')
        
    # Initialize position 
    X    = init_position(lb, ub, N, dim)
    
    # Binary conversion
    Xbin = binary_conversion(X, thres, N, dim)
    
    # Fitness at first iteration
    fit  = np.zeros([N, 1], dtype='float')
    Xgb  = np.zeros([1, dim], dtype='float')
    fitG = float('inf')
    Xglobal  = np.zeros([1, dim], dtype='float')

    for i in range(N):
        fit[i,0] = Fun(xtrain, ytrain, Xbin[i,:], opts)
        if fit[i,0] < fitG:
            Xgb[0,:] = X[i,:]
            fitG     = fit[i,0]
        
    # Pre
    curve = np.zeros([1, max_iter], dtype='float') 
    t     = 0
    
    curve[0,t] = fitG.copy()
    print("Generation:", t + 1)
    print("Best (MFRO):", curve[0,t])
    t += 1

    while t < max_iter:
        
        for i in range(N):
            # Cyclone foraging (Eq. 5, 6, 7)
            for d in range(dim):
                if np.random.rand() < 0.5:
                    r1 = np.random.uniform()
                    beta = 2 * np.exp(r1 * (max_iter + 1 -  t) / (max_iter+1)) * np.sin(2 * np.pi * r1)

                    if (t + 1) / (max_iter+1) < np.random.rand():
                        k = np.random.uniform(lb[0,d], ub[0,d])
                        if i == 0:
                            X[i,d] = k + np.random.uniform() * (k - X[i,d]) + \
                                beta * (k - X[i,d])
                        else:
                            X[i,d] = k + np.random.uniform() * (X[i-1,d] - X[i,d]) + \
                                beta * (k - X[i,d])
                    else:
                        if i == 0:
                            X[i,d] =Xgb[0,d] + np.random.uniform() * (Xgb[0,d] - X[i,d]) + \
                                beta * (Xgb[0,d] - X[i,d])
                        else:
                            X[i,d] =Xgb[0,d] + np.random.uniform() * (X[i-1,d] - X[i,d]) + \
                                beta * (Xgb[0,d] - X[i,d])
                # Chain foraging (Eq. 1,2)
                else:
                    r = np.random.uniform()
                    alpha = 2 * r * np.sqrt(np.abs(np.log(r)))
                    if i == 0:
                        X[i,d] = X[i,d] + r * (Xgb[0,d] - X[i,d]) + \
                            alpha * (Xgb[0,d] - X[i,d])
                    else:
                        X[i,d] = X[i,d] + r * (X[i-1,d] - X[i,d]) + \
                            alpha * (Xgb[0,d] - X[i,d])

                X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d])
        
        for i in range(N):
            for d in range(dim):
            # Somersault foraging   (Eq. 8)
                X[i,d] = X[i,d] + somersault_range * \
                    (np.random.uniform() * Xgb[0, d] - np.random.uniform() * X[i][d])
                X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d])
                
        
        #feedbackMechansim
        if( (Xgb[0,:] == Xglobal[0,:]).all() ):
            dyn_feedback+=1
        else:
            dyn_feedback=0
            Xglobal = Xgb.copy()
        
        
        if dyn_feedback >= feedback_max:
            print("---------------------------------------")
            #dyn_feedback=0
            
            idx_list = np.random.choice(range(0, N), n_changes, replace=False)
            #initialize position 
            X_child  = init_position(lb, ub, n_changes, dim)
            #nfe_epoch += self.n_changes
            for idx_counter, idx in enumerate(idx_list):
                #print(idx_counter, idx)
                for d in range(dim):
                    X[idx_counter,d] = X_child[idx_counter,d]
   
               
        # Binary conversion
        Xbin = binary_conversion(X, thres, N, dim)
        
        # Fitness
        for i in range(N):
            fit[i,0] = Fun(xtrain, ytrain, Xbin[i,:], opts)
            if fit[i,0] < fitG:
                Xgb[0,:] = X[i,:]
                fitG     = fit[i,0]
        
        # Store result
        curve[0,t] = fitG.copy()
        print("Generation:", t + 1)
        print("Best (MRFO):", curve[0,t])
        t += 1            

            
    # Best feature subset
    Gbin       = binary_conversion(Xgb, thres, 1, dim) 
    Gbin       = Gbin.reshape(dim)    
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    # Create dictionary
    woa_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}
    
    return woa_data 