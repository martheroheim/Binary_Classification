import json
import numpy as np


class Weights: 
    
    def __init__(self, config_file):
        with open(config_file, 'r') as config:
            conf = json.load(config)

        parameters = conf['parameters']
        
        self.K = parameters['K']
        self.tau = parameters['TAU']
        self.h = parameters['H']
        self.d = parameters['D']
        self.num = parameters['NUM_POINTS']
        self.my = parameters['MY']
        
        self.Y = np.ones((self.K,self.d,self.num)) 
        self.W = np.random.randn(self.K, self.d, self.d) 
        self.b = np.random.randn(self.K, self.d, 1)
        self.P = np.ones((self.K,self.d,self.num)) 
        self.c = np.ones((self.num,1))
        self.w = np.ones((self.d,1))

    def addMatVec(self, i):
        return ((self.W[i,:,:]@self.Y[i,:,:]) + self.b[i,:])
    
    def yNext(self, i): 
        return self.Y[i,:,:] + self.h*np.tanh(self.addMatVec())
    
    def Z(self):
        return eta((self.Y[(self.K-1),:,:].T)@self.w + self.my) 
    
    '''ForewardFunction: plotting the result of the spiral'''
    def Z0(self, grid): 
        self.num,self.d=np.shape(grid)
        self.Y=np.zeros((self.K+1,self.num,self.d))
        self.Y[0]=grid
        for k in range(1,self.K+1):
            self.Y[k,:,:] = self.Y[k-1,:,:] + self.h*np.tanh(self.addMatVec(k-1))
        return eta((Y[-1].T)@w + my)
    
    def pPrev(self, k):
        return self.P[k,:,:] + self.h * (self.W[(k-1),:,:].T)@np.multiply(dsig(self.addMatVec(k-1)),self.P[k,:,:])
    
    def find_gradient(self,U):
        """
        Finds the gradient of the different arrays or 
        numbers in U 
        Input: 
            U: an numpy array on the form [W, b, w, my]
        Output: 
            a numpy array with the gradients of the input 
            on the form [dW, db, dw, dmy]
        """

        dW = self.h * (self.P * dsig(U[0] @ self.Y + U[1])) @ (self.Y.transpose((0, 2, 1))) 
        db = self.h * (self.P * dsig(U[0] @ self.Y + U[1])) @ np.ones((self.num,1))
        dmy = np.transpose(deta(np.transpose(self.Y[(self.K - 1),:,:]) @ U[2] + U[3])) @ (self.Z()-self.c)       
        dw = self.Y[self.K-1,:,:] @ ((self.Z()-self.c) * deta((self.Y[self.K-1,:,:].T) @ U[2] + U[3]))      

        return np.array([dW, db, dw, dmy])
    
    def vanilla(self):
        """
        Updates U with the find_gradient 
        function and a learningrate; tau
        input: 
            W: a np array with dimensions K,d,d
            b: a np array with dimensions K,d,1
            w: a np array with dimensions d,1 
            (uprise vector)
            my: a float
        output: 
            U: a numpy array on the form [W, b, w, my]
        """
        U = np.array([self.W, self.b, self.w, self.my])
        
        gU = self.find_gradient(U)
        
        U = U - self.tau*gU
        
        self.W = U[0]
        self.b = U[1]
        self.w = U[2]
        self.my = U[3]
        
    def adam(X, dX, j, m, v, beta_1, beta_2, alpha, epsilon):
        """
        NOT FINISHED
        """
        beta_1=0.9
        beta_2=0.999
        alpha = 0.01            #steglengde
        epsilon = 1e-8

        g = dX

        m = beta_1 * m + (1 - beta_1) * g     
        v = beta_2 * v + (1 - beta_2) * (g*g)
        m_hat = m / (1 - beta_1**j)
        v_hat = v / (1 - beta_2**j)
        for iu in range(len(v_hat)):         #tar sqrt(v_hat) elementvis
            v_hat[iu] = np.sqrt(v_hat[iu])
        X = X - alpha * m_hat / (v_hat + epsilon)
        return X, m, v

def eta(x): 
    return (np.exp(x))/(np.exp(x) + 1)

def deta(x):
    return np.exp(x)/(np.exp(x)+1)**2

def dsig(x):
    return 4/(np.exp(x)+np.exp(-x))**2