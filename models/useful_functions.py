import numpy as np



def shift(x: np.array, n:float): 
    if n >0 : 
        return np.concatenate((np.array([ x[0] for i in range(int(n))]), x))[:len(x)] # we assume that the n first values are the same as the first value of the array
    elif n < 0 :
        return np.concatenate((x, np.array([ x[-1] for i in range(int(-n))])))[-len(x):]
    else :
        return x
    

