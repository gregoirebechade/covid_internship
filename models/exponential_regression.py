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

# remove a columns from a df: 
df.drop(columns=['Unnamed: 0'], inplace=True)
new_deaths=np.array(df['new_deaths'])
death_cumul=np.array([sum(new_deaths[:i]) for i in range(len(new_deaths))])
dates_of_pandemic=np.arange(len(new_deaths))


def exponential_func(x, a, b, c):
    return a*np.exp(b*(x))+c
# IC with the formula in paper 3: 

def h(theta, x_i):
    return theta[0]*np.exp(theta[1]*(x_i))+theta[2]

def grad_theta(theta, x_i): 
    d_theta=0.0001
    grad=np.zeros(len(theta))
    for i in range(len(theta)): 
        theta_plus=theta.copy()
        theta_plus[i]+=d_theta
        grad[i]=(h(theta_plus,x_i)-h(theta,x_i))/d_theta
    return grad


def compute_A(theta, X): 
    A=np.zeros((len(X), len(theta)))
    for i in range(len(X)): 
        A[i]=grad_theta(theta, X[i])
    return A


def estimate_sigma2(data, prediction,d): 
    return np.sum((data-prediction)**2)/(len(data)-d)
def objective_function(data, theta, X): 
    return 0.5*np.sum((data-h(theta, X))**2)



def hessian_obj_function(data, theta, X): 
    d_theta=0.0001
    hessian=np.zeros((len(theta), len(theta)))
    for i in range(len(theta)): 
        for j in range(len(theta)):
            theta_plus_i=theta.copy()
            theta_plus_j=theta.copy()
            theta_plus_ij=theta.copy()
            theta_plus_i[i]+=d_theta
            theta_plus_j[j]+=d_theta
            theta_plus_ij[i]+=d_theta
            theta_plus_ij[j]+=d_theta
            hessian[i,j]=(objective_function(data, theta_plus_ij, X)-objective_function(data, theta_plus_i, X)-objective_function(data, theta_plus_j, X)+objective_function(data, theta, X))/(d_theta**2)
    return hessian


def f_for_delta_method(train_dates, data, interval): 
    theta, _ = curve_fit(exponential_func, train_dates[interval], data[interval], p0=[ 1.33991316e+01 , 1.21453531e-01,  -1.92062731e+02], maxfev = 10000)
    return theta

def grad_f_for_delta_method(train_dates, data, interval): 
    d_n=0.1
    grad=np.zeros(( len(data), 3) )
    for i in range(len(data)): 
        data_plus=data.copy()
        data_plus[i]+=d_n
        grad[i]=(f_for_delta_method(train_dates, data, interval)-f_for_delta_method(train_dates, data, interval))/d_n
    return grad



def grad_theta_h(theta, x): 
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
        self.data=data
        train_dates=np.array(train_dates)
        self.train_dates=train_dates
        min=len(data)-30
        max=len(data)-1
        interval=[i for i in range(min,max)]
        self.interval=interval
        self.p, self.cov =curve_fit(exponential_func, train_dates[interval], data[interval], p0=[ 1,1,1], maxfev = 1000000)
        self.trained=True


    def predict(self, reach, alpha, method='covariance'):
        assert self.trained, 'The model has not been trained yet'
        a=self.p[0]
        b=self.p[1]
        c=self.p[2]
        window_prediction=np.array([i for i in range(len(self.train_dates), len(self.train_dates) + reach )])
        self.window_prediction=window_prediction
      
        prediction=exponential_func(window_prediction,a,b,c)
        self.prediction=prediction

        if method == 'covariance': # we implemented four methods to compute the confidence intervals
            print('covariance method')
            perr = np.sqrt(np.diag(self.cov))
            self.perr=perr
            
        elif method == 'estimate_sigma': 
            print('estimate sigma')
            sigma2=estimate_sigma2(self.data[self.interval], exponential_func(self.train_dates[self.interval], *self.p), 4)
            A=compute_A(self.p, self.train_dates[self.interval])
            try : 
                cov=sigma2*np.linalg.inv(np.matmul(A.transpose(), A))/len(self.interval)
            except LinAlgError: 
                hessian += np.eye(hessian.shape[0]) * 1e-5
                cov = np.linalg.inv(hessian)
            perr=np.sqrt(np.diag(cov))
            self.cov=cov
            self.perr=perr
        elif method == 'hessian':
            print('hessian')
            hessian=hessian_obj_function(self.data[self.interval], self.p, self.train_dates[self.interval])
            self.hess=hessian
            try: 
                cov=np.linalg.inv(hessian)
            except LinAlgError:
                hessian += np.eye(hessian.shape[0]) * 1e-5
                cov = np.linalg.pinv(hessian)
            perr=np.sqrt(abs(np.diag(cov)))
            self.cov=cov
            self.perr=perr
        elif method == 'delta': 
            print('delta')
            sigma2=estimate_sigma2(self.data[self.interval], exponential_func(self.train_dates[self.interval], *self.p), 3) * np.identity(len(self.data))
            grad=grad_f_for_delta_method(self.train_dates, self.data, self.interval)
            perr = np.sqrt(np.diag(np.matmul(np.matmul(grad.transpose(), sigma2) , grad)))
            self.perr = perr
            
        intervals=[prediction]
        a_sampled=[]
        b_sampled=[]
        c_sampled=[]
        for i in range(100): 
            # a_r= np.random.normal(self.p[0], perr[0], 1)[0]
            # b_r=np.random.normal(self.p[1], perr[1], 1)[0]
            # c_r=np.random.normal(self.p[2], perr[2], 1)[0]
            a_r, b_r, c_r = np.random.multivariate_normal(self.p, self.cov)
            a_sampled.append(a_r)
            b_sampled.append(b_r)
            c_sampled.append(c_r)
            prediction_sampled=exponential_func(window_prediction,a_r, b_r,c_r)
            intervals.append(prediction_sampled)
        self.a_sampled=a_sampled
        self.b_sampled=b_sampled
        self.c_sampled=c_sampled
        intervals=np.array(intervals).transpose()
        self.intervals=intervals
        ci_low=np.array([np.quantile(intervals[i], alpha/2) for i in range(reach)])
        ci_high=np.array([np.quantile(intervals[i],1-alpha/2) for i in range(reach)])


        ##############################
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



class MultiDimensionalExponentialRegression(Multi_Dimensional_Model): 
    def train(self, train_dates, data):
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
        self.trained=True


    def predict(self, reach, alpha, method='covariance'):
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

        perr = np.sqrt(np.diag(self.cov))
        self.perr=perr     
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


