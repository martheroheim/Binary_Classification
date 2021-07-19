import json
import numpy as np

def matrics_n_vec(config_file):
    """
    Make the matrix and vetors
    input:  
        p: dictionary from config-file
    output: 
        matrix and vectors
    """
    
    
    with open(config_file, 'r') as config:
        conf = json.load(config)
        
    parameters = conf['parameters']
    
    K = parameters['K']
    tau = parameters['TAU']
    h = parameters['H']
    d = parameters['D']
    num = parameters['NUM_POINTS']
    my = parameters['MY']
    
    Y = np.ones((K,d,num)) 
    W = np.random.randn(K, d, d) 
    b = np.random.randn(K, d, 1) 
    P = np.ones((K,d,num)) 
    c = np.ones((num,1))
    w = np.ones((d,1)).fill(3) 
    
    mat_n_vec = {"Y" : Y, 
                "W" : W,
                "b" : b, 
                "P" : P, 
                "c" : c, 
                "w" : w, 
                "num" : num,
                "my" : my, 
                "tau" : tau}
    return mat_n_vec

if __name__ == '__main__':
    mat = matrics_n_vec("../data/config_spiral.json")
    print("\n------CHECK-----\nCheck that every matrix and vector are the right shapes:")
    for key, value in mat.items(): 
        print(key, np.shape(value))