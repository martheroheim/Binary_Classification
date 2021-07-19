import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
from spirals import get_data_spiral_2d
from plotting import plot_progression 
from IPython.display import Markdown, display

def printmd(string):
    display(Markdown(string)) #For å kunne skrive ut i fet tekst
    
    
def addMatVec(W, Y, b, k):
    return ((W[k,:,:]@Y[k,:,:]) + b[k,:])


def yNext(Y, W, b, k): 
       return Y[k,:,:] + h*np.tanh(addMatVec(W,Y,b,k))

def Z(Y):
    return eta((Y[(K-1),:,:].T)@w + my) #skal her bruke den siste Y_{k}, K !

'''ForewardFunction: plotting the result of the spiral'''
def Z0(grid): 
    I,d=np.shape(grid)
    Y=np.zeros((K+1,I,d))
    Y[0]=grid
    for k in range(1,K+1):
        Y[k,:,:] =Y[k-1,:,:] + h*np.tanh(addMatVec(W,Y,b,k-1))
    return eta((Y[-1].T)@w + my)

'''lastFunction: used for plotting the last picture of the spiral'''
def Z_last(Y): 
    return eta((Y.T)@w + my)
    
def eta(x): 
    return (np.exp(x))/(np.exp(x) + 1)

def deta(x):
    return np.exp(x)/(np.exp(x)+1)**2

def dsig(x):
    return 4/(np.exp(x)+np.exp(-x))**2

def pPrev(P, k):
    return P[k,:,:] + h*(W[(k-1),:,:].T)@np.multiply(dsig(addMatVec(W,Y,b,k-1)),P[k,:,:])


def find_gradient(U):
    """
    Finds the gradient of the different arrays or 
    numbers in U 
    Input: 
        U: an numpy array on the form [W, b, w, my]
    Output: 
        a numpy array with the gradients of the input 
        on the form [dW, db, dw, dmy]
    """
    
    dW = h*(P*dsig(U[0]@Y+U[1]))@(Y.transpose((0, 2, 1))) 
    db = h*(P*dsig(U[0]@Y+U[1]))@np.ones((I,1))
    dmy = np.transpose(deta(np.transpose(Y[(K-1),:,:])@U[2] + U[3]))@(Z(Y)-c)             
    dw = Y[K-1,:,:]@((Z(Y)-c)*deta((Y[K-1,:,:].T)@U[2] + U[3]))      
    
    return np.array([dW, db, dw, dmy])

def vanilla(W,b,w,my):
    """
    Updates U with the find_gradient 
    function and a learningrate; tau
    input: 
        W: a np array with dimensions K,d,d
        b: a np array with dimensions K,d,1
        w: a np array with dimensions d,1 (uprise vector)
        my: a float
    output: 
        U: a numpy array on the form [W, b, w, my]
    """
    U = np.array([W, b, w, my])
    gU = find_gradient(U)
    U = U - tau*gU
    return U


def adam(W,b,w,my,j, m, v):
    U = np.array([W, b, w, my])
    g = find_gradient(U) #Tar U, returnerer oppdatert U

    beta_1=0.9
    beta_2=0.999
    alpha = 0.01            #steglengde  
    epsilon = 1e-8        
    g = find_gradient(U)
    m = beta_1 * m + (1 - beta_1) * g     
    v = beta_2 * v + (1 - beta_2) * (g*g)
    m_hat = m / (1 - beta_1**j)
    v_hat = v / (1 - beta_2**j)
    for i in range(len(v_hat)): #tar sqrt(v_hat)
        v_hat[i] = np.sqrt(v_hat[i])
    U = U - alpha * m_hat / (v_hat + epsilon)
    return U, m, v #returnerer m og v for å oppdatere disse