import numpy as np

def innerproduct(X,Z=None):
    # function innerproduct(X,Z)
    #
    # Computes the inner-product matrix.
    # Syntax:
    # D=innerproduct(X,Z)
    # Input:
    # X: nxd data matrix with n vectors (rows) of dimensionality d
    # Z: mxd data matrix with m vectors (rows) of dimensionality d
    #
    # Output:
    # Matrix G of size nxm
    # G[i,j] is the inner-product between vectors X[i,:] and Z[j,:]
    #
    # call with only one input:
    # innerproduct(X)=innerproduct(X,X)
    #
    if Z is None: # case when there is only one input (X)
        G=innerproduct(X,X)
    else:  # case when there are two inputs (X,Z)
        G=np.dot(X,Z.T)
    return G

def l2distance(X,Z=None):
    # function D=l2distance(X,Z)
    #
    # Computes the Euclidean distance matrix.
    # Syntax:
    # D=l2distance(X,Z)
    # Input:
    # X: nxd data matrix with n vectors (rows) of dimensionality d
    # Z: mxd data matrix with m vectors (rows) of dimensionality d
    #
    # Output:
    # Matrix D of size nxm
    # D(i,j) is the Euclidean distance of X(i,:) and Z(j,:)
    #
    # call with only one input:
    # l2distance(X)=l2distance(X,X)
    #

    if Z is None:
        D=l2distance(X,X)
    else:  # case when there are two inputs (X,Z)
        G=innerproduct(X,Z)
        n=X.shape[0]
        m=Z.shape[0]
        d=X.shape[1]
        Xshape = X.shape
        Zshape = Z.shape
        X_ones = np.ones((1,d))
        Z_ones = np.ones((1,d))
        xi = np.transpose(np.dot(X_ones,np.transpose(np.square(X))))
        zj = np.dot(Z_ones, np.transpose(np.square(Z)))
        S=np.zeros((n,m))+xi
        R=np.zeros((n,m))+zj
        D2=abs(S-2*G+R)
        D=np.sqrt(D2)
    return D

def insertNeighbor(j, d, ind, dis):
    newDis=dis
    newInd=ind
    for x in range(len(dis)):
        if d <= dis[x]:
            newDis = dis[:x]+[d]+dis[x:len(dis)-1]
            newInd = ind[:x]+[j]+ind[x:len(dis)-1]
            break
            
    return (newInd, newDis)

def findknn(xTr,xTe,k):
    """
    function [indices,dists]=findknn(xTr,xTe,k);
    
    Finds the k nearest neighbors of xTe in xTr.
    
    Input:
    xTr = nxd input matrix with n row-vectors of dimensionality d
    xTe = mxd input matrix with m row-vectors of dimensionality d
    k = number of nearest neighbors to be found
    
    Output:
    indices = kxm matrix, where indices(i,j) is the i^th nearest neighbor of xTe(j,:)
    dists = Euclidean distances to the respective nearest neighbors
    """
    if k > len(xTr):
        k = len(xTr)
        
    D=l2distance(xTe, xTr)
    (m,n) = D.shape
    
    indices = []
    dists = []
    for i in range(m):
        smallest_indices = np.argsort(D[i])
        inds = smallest_indices[:k]
        dis = D[i,smallest_indices[:k]]
        indices.append(inds)
        dists.append(dis)

    indices = np.transpose(np.array(indices))
    dists = np.transpose(np.array(dists))
    return indices, dists

#<GRADED>
def mode(y):
    if len(y) == 1:
        return y[0]
    else:
        counts = {}
        for i in range(len(y)):
            if y[i] in counts:
                counts[y[i]] +=1
            else:
                counts[y[i]] =1
        m = max(counts, key=counts.get)
        c = counts[m]
        del counts[m]
        if len(counts) == 0:
            return m
        else:
            second_highest = max(counts, key=counts.get)

            if counts[second_highest] == c:
                return mode(y[:(len(y)-1)])
            else:
                return m

def knnclassifier(xTr,yTr,xTe,k):
    """
    function preds=knnclassifier(xTr,yTr,xTe,k);
    
    k-nn classifier 
    
    Input:
    xTr = nxd input matrix with n row-vectors of dimensionality d
    xTe = mxd input matrix with m row-vectors of dimensionality d
    k = number of nearest neighbors to be found
    
    Output:
    
    preds = predicted labels, ie preds(i) is the predicted label of xTe(i,:)
    """
    tup = findknn(xTr, xTe, k)
    I = np.transpose(tup[0])
    D = np.transpose(tup[1])
    preds = []
    for i in range(len(xTe)):
        inds = I[i]
        y = np.ndarray.flatten(yTr[inds])
        m = mode(y)
        preds.append(m)
        
    preds = np.ndarray.flatten(np.array(preds))
    return preds
