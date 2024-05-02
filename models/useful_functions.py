import numpy as np



def shift(x: np.array, n:float): 
    return np.concatenate((np.array([ x[0] for i in range(int(n))]), x))[:len(x)] # we assume that the n first values are the same as the first value of the array

