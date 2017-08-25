# An implementation of Adagrad (Gradient Descent) and logistic loss function

import numpy as np

def adagrad(func,w,alpha,maxiter,delta=1e-02):
    """
    INPUT:
    func    : function to minimize
              (loss, gradient = func(w))
    w       : d dimensional initial weight vector 
    alpha   : initial gradient descent stepsize (scalar)
    maxiter : maximum amount of iterations (scalar)
    delta   : if norm(gradient)<delta, it quits (scalar)
    
    OUTPUTS:
     
    w      : d dimensional final weight vector
    losses : vector containing loss at each iteration
    """
    losses = np.zeros(maxiter)
    eps = 1e-06
    
    ## fill in your code here
    d = w.shape[0]
    z = np.zeros(d)
    for i in range(len(losses)):
        loss, gradient = func(w)
        z = z + np.square(gradient)
        w_tone = (w - alpha*gradient/np.sqrt(z+eps))
        losses[i] = loss
        w = w_tone
        if np.linalg.norm(gradient) < delta:
            break
    return w, losses

def logistic(w,xTr,yTr):
    """
    INPUT:
    w     : d   dimensional weight vector
    xTr   : nxd dimensional matrix (each row is an input vector)
    yTr   : n   dimensional vector (each entry is a label)
    
    OUTPUTS:
    loss     : the total loss obtained with w on xTr and yTr (scalar)
    gradient : d dimensional gradient at w
    """
    n, d = xTr.shape
    
    ## fill in your code here
    R = -yTr*np.dot(w,xTr.T)
    v = np.log(1.+np.exp(R))
    loss = sum(v)
    y = yTr.reshape((n,1))
    numerator = y*xTr
    denominator = 1. + np.exp(yTr*w.dot(xTr.T))
    gradient = -numerator.T.dot(1./denominator)
    return loss, gradient

