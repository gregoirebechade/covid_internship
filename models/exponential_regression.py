import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import sys 
sys.path.append('./models/')
from Model import Model, Multi_Dimensional_Model 
df = pd.read_csv('deaths_and_infections.csv')
from numpy.linalg import LinAlgError
import scipy.stats



def exponential_func(x, a, b, c):
    return a*np.exp(b*(x))+c

def h(theta, x_i):# theta represent the parameters of the model
    return theta[0]*np.exp(theta[1]*(x_i))+theta[2]



def grad_theta_h(theta, x): # function to compute the gradient of h (which represent the prediction function) with respect to theta, the parameters of the model
    a=theta[0]
    b=theta[1]
    c=theta[2]
    grad=np.zeros(3)
    grad[0]=np.exp(b*x) 
    grad[1]=a*x*np.exp(b*x)
    grad[2]=1
    return grad 

class ExponentialRegression(Model): 
    def train(self, train_dates, data):
        self.name='Exponential regression'
        self.data=data
        train_dates=np.array(train_dates)
        self.train_dates=train_dates
        min=len(data)-15
        max=len(data)-1
        interval=[i for i in range(min,max)]
        self.interval=interval
        self.p, self.cov =curve_fit(exponential_func, train_dates[interval], data[interval], p0=[ 1,1,1], maxfev = 1000000)
        self.trained=True


    def predict(self, reach, alpha):
        assert self.trained, 'The model has not been trained yet'
        a=self.p[0]
        b=self.p[1]
        c=self.p[2]
        window_prediction=np.array([i for i in range(len(self.train_dates), len(self.train_dates) + reach )])
        self.window_prediction=window_prediction
      
        prediction=exponential_func(window_prediction,a,b,c)
        self.prediction=prediction

        ci_low=[]
        ci_high=[]
        grads= []
        vars=[]
        for i in range(len(prediction)):
            index = self.window_prediction[i] 
            grad=grad_theta_h(self.p, index)
            grads.append(grad)
            varhtheta=self.cov 
            varprediction=np.matmul(np.matmul(grad.transpose(), varhtheta), grad)
            vars.append(varprediction)
            down = scipy.stats.norm.ppf(alpha/2, loc=prediction[i], scale=np.sqrt(varprediction))
            ci_low.append(down)
            up = scipy.stats.norm.ppf(1-(alpha/2), loc=prediction[i], scale=np.sqrt(varprediction))
            ci_high.append(up)
        self.ci_low=ci_low
        self.ci_high=ci_high
        self.grads=grads
        self.vars=vars
        return prediction, [ci_low, ci_high]




def exponential_function_m(X, a, b,c, d, e): 
    i, n_infected, mobility = X

    return a * np.exp(b * mobility + c * i+ d * n_infected) +e



def grad_theta_h_m(theta, x): 
    a=theta[0]
    b=theta[1]
    c=theta[2]
    d=theta[3]
    e=theta[4]
    grad=np.zeros(5)
    i, n_infected, mobility = x
    grad[0]=np.exp(b * mobility +c* i+ d * n_infected)
    grad[1]=a * np.exp(b * mobility +c* i+ d * n_infected) * mobility
    grad[2]=a * np.exp(b * mobility +c* i+ d * n_infected) * i
    grad[3]=a * np.exp(b * mobility +c* i+ d * n_infected) * n_infected
    grad[4]=1
    return grad 


def shift(x: np.array, n:float): 
    return np.concatenate((np.array([ x[0] for i in range(int(n))]), x))[:len(x)] # we assume that the n first values are the same as the first value of the array

def intermediate(x: np.array, a, b, c, d, e, shift1, shift2, n_infected_normalized, mobility): 
    res=[]
    for elt in x: 
        uno=((shift1 - int(shift1)))*exponential_function_m([elt, n_infected_normalized[int(elt-shift1)], mobility[int(elt-shift2)]], a, b, c, d, e) + (1-(shift1 - int(shift1)))*exponential_function_m([elt, n_infected_normalized[int(elt-shift1)+1], mobility[int(elt-shift2)]], a, b, c, d, e)
        dos=((shift2 - int(shift2)))*exponential_function_m([elt, n_infected_normalized[int(elt-shift1)], mobility[int(elt-shift2)]], a, b, c, d, e) + (1-(shift2 - int(shift2)))*exponential_function_m([elt, n_infected_normalized[int(elt-shift1)], mobility[int(elt-shift2)+1]], a, b, c, d, e)
        res.append(0.5*uno + 0.5 * dos)
    return res



class MultiDimensionalExponentialRegression(Multi_Dimensional_Model): 

    def train(self, train_dates, data):
        self.name='multi exponential regression'
        self.data=data
        maxi=np.max(data[1])
        self.n_infected_normalized=np.array([i/maxi for i in data[1]]) # to avoid too big values in the exponential function
        n_infected_normalized=self.n_infected_normalized
        train_dates=np.array(train_dates)
        self.train_dates=train_dates
        min=len(data[0])-30
        max=len(data[0])-1
        interval=[i for i in range(min,max)]
        self.interval=interval
        
        self.p, self.cov =curve_fit(exponential_function_m, (train_dates[interval], n_infected_normalized[interval], data[2][interval]),data[0][interval],  p0=[ 1,1, 1, 1,1], maxfev = 1000000)
        grid_search=True
        if grid_search : # we add a shift parameter to the model
            lossmin=np.inf
            pmin=None
            covmin =None
            shift1min = None
            shift2min = None
            for shift1 in range( 15): 
                for shift2 in range(15): 
                    interval1=[i - shift1 for i in interval]
                    interval2= [ i-shift2 for i in interval]
                    try : 
                        p, cov =curve_fit(exponential_function_m, (train_dates[interval], n_infected_normalized[interval1], data[2][interval2]),data[0][interval],  p0=[ 1,1, 1, 1,1], maxfev = 10000)
                        loss=np.sum((np.array([exponential_function_m([train_dates[interval][i],n_infected_normalized[interval1][i],data[2][interval2][i] ], p[0], p[1], p[2], p[3], p[4])  for i in range(len(interval))]) - np.array(data[0][interval]))**2)
                        if loss < lossmin: 
                            lossmin = loss
                            pmin = p
                            covmin = cov
                            shift1min = shift1
                            shift2min = shift2
                    except RuntimeError: 
                        print('RuntimeError')
            self.p = pmin
            self. cov = covmin
            self.shift1 = shift1min
            self.shift2=shift2min
            
        self.trained=True


    def predict(self, reach, alpha):
        assert self.trained, 'The model has not been trained yet'
        a=self.p[0]
        b=self.p[1]
        c=self.p[2]
        d=self.p[3]
        e=self.p[4]
        window_prediction=np.array([i for i in range(len(self.train_dates), len(self.train_dates) + reach )])
        self.window_prediction=window_prediction
        last_value_of_mobility=self.data[2][-1]
        last_value_of_infected=self.n_infected_normalized[-1]
        prediction_interval=np.array([window_prediction, np.array([last_value_of_infected for i in range(len(window_prediction))]), np.array([last_value_of_mobility for i in range(len(window_prediction))])]) 
        prediction=exponential_function_m(prediction_interval,a,b,c,d,e)
        self.prediction=prediction
        ci_low=[]
        ci_high=[]
        grads= []
        vars=[]
        for i in range(len(prediction)):
            index = self.window_prediction[i] 
            n_infected=last_value_of_infected
            mobility=last_value_of_mobility
            grad=grad_theta_h_m(self.p, [index, n_infected, mobility])
            grads.append(grad)
            varhtheta=self.cov[:5, :5]
            varprediction=np.matmul(np.matmul(grad.transpose(), varhtheta), grad)
            vars.append(varprediction)
            down = scipy.stats.norm.ppf(alpha/2, loc=prediction[i], scale=np.sqrt(varprediction))
            ci_low.append(down)
            up = scipy.stats.norm.ppf(1-(alpha/2), loc=prediction[i], scale=np.sqrt(varprediction))
            ci_high.append(up)
        self.ci_low=ci_low
        self.ci_high=ci_high
        self.grads=grads
        self.vars=vars
        return prediction, [ci_low, ci_high]


