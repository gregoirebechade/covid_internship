from Model import Model
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import sys
sys.path.append('./../')


df = pd.read_csv('deaths_and_infections.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)
new_deaths=np.array(df['new_deaths'])
death_cumul=np.array([sum(new_deaths[:i]) for i in range(len(new_deaths))])
dates_of_pandemic=np.arange(len(new_deaths))

s_0=1000000 -1
i_0=1
r_0=0
d_0=0
t=len(death_cumul-1)
dt=0.001


def derive(x, beta, N, gamma, d):
    s=x[0]
    i=x[1]
    r=x[2]
    deads=x[3]
    return np.array([-beta*s*i/N, beta*s*i/N - gamma*i - d * i , gamma*i , d * i ])



def run_sir(x0, beta, gamma,d,  t, dt):
    x=x0
    S=[x[0]]
    I=[x[1]]
    R=[x[2]]
    D=[x[3]] # deads 
    n_iter=int(t/dt)
    N=sum(x0)
    for i in range(n_iter):
        x=x+dt*derive(x, beta, N, gamma, d)
        S.append(x[0])
        I.append(x[1])
        R.append(x[2])
        D.append(x[3])
    s_final=[]
    i_final=[]
    r_final=[]
    d_final=[]
    time=np.linspace(0, t, int(t/dt) )
    for i in range(len(time)-1):
        if abs(time[i]-int(time[i]))<dt: 
            s_final.append(S[i])
            i_final.append(I[i])
            r_final.append(R[i])
            d_final.append(D[i])
    return s_final, i_final, r_final, d_final
    



def differenciate(x): 
    dx=[x[i+1]-x[i] for i in range(len(x)-1)]
    return dx

def sir_for_optim(x, beta, gamma, d):
    # x is a list of dates (0 - 122)
    x0=[s_0, i_0, r_0, d_0]
    t=len(x)
    S,I,R,D=run_sir(x0, beta, gamma,d,  t, dt)
    zer=np.array([0])
    d_arr=np.array(D)
    return differenciate(np.concatenate((zer,d_arr))) # returns a value per day


def objective_function(data, theta, X):
    deads = sir_for_optim(X, theta[0], theta[1], theta[2])
    return np.sum((data-deads)**2)/(2)

def hessian_objective_function(data, theta, X): 
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


def estimate_sigma2(data, prediction,d): 
    return np.sum((data-prediction)**2)/(len(data)-d)




def f_for_delta_method(train_dates, data): 

    theta,cov= curve_fit(sir_for_optim, train_dates,data, p0=[ 5.477e-01 , 2.555e-02 , 5.523e-04],  bounds=([0,0,0], [5,5,5]))

    return theta

def grad_f_for_delta_method(train_dates, data): 
    d_n=0.1
    grad=np.zeros(( len(data), 3) )
    for i in range(len(data)): 
        data_plus=data.copy()
        data_plus[i]+=d_n
        grad[i]=(f_for_delta_method(train_dates, data)-f_for_delta_method(train_dates, data))/d_n
    return grad


class SIRD_model(Model): 
    s_0=1000000 -1
    i_0=1
    r_0=0
    d_0=0
    dt=0.001
    def train(self, train_dates, data):
        self.data=data
        self.train_dates=train_dates
        p,cov= curve_fit(sir_for_optim, self.train_dates,data, p0=[ 5.477e-01 , 2.555e-02 , 5.523e-04],  bounds=([0,0,0], [5,5,5]))
        self.beta=p[0]
        self.gamma=p[1]
        self.d=p[2]
        self.cov=cov
        self.trained= True


    def predict(self, reach, alpha, method='covariance'):
        assert self.trained, 'The model has not been trained yet'
        deads=sir_for_optim(np.array([i for i in range(len(self.train_dates)+reach)]), self.beta, self.gamma,self.d)
        prediction =  deads[-reach:]
        if method == 'covariance': 
            perr = np.sqrt(np.diag(self.cov)) # Idea from: https://github.com/philipgerlee/Predicting-regional-COVID-19-hospital-admissions-in-Sweden-using-mobility-data.
        elif method == 'hessian':
            print('hessian')
            hessian=hessian_objective_function(self.data, [self.beta, self. gamma, self.d], self.train_dates)
            self.hess=hessian
            cov=np.linalg.inv(hessian)
            perr=np.sqrt(abs(np.diag(cov)))
            self.perr=perr
        elif method == 'delta': 
            print('delta')
            p=[self.beta, self.gamma, self.d]
            sigma2=estimate_sigma2(self.data, deads[:-reach], len(p)) * np.identity(len(self.data))
            grad=grad_f_for_delta_method(self.train_dates, self.data)
            perr = np.sqrt(np.diag(np.matmul(np.matmul(grad.transpose(), sigma2) , grad)))
            self.perr = perr

        intervals=[prediction]
        beta_sampled=[]
        gamma_sampled=[]
        d_sampled=[]
        for i in range(100): 
            # beta_r=max(0, np.random.normal(self.beta, perr[0], 1)[0])
            # gamma_r=max(0,np.random.normal(self.gamma, perr[1], 1)[0])
            # d_r=max(0,np.random.normal(self.d, perr[2], 1)[0])
            a=np.random.multivariate_normal([self.beta,self.gamma,self.d], self.cov, 1)[0]
            # a[1]=abs(a[1])
            # while not (a>0).all(): 
            #     a=np.random.multivariate_normal([self.beta,self.gamma,self.d], self.cov, 1)[0]
            #     a[1]=abs(a[1])
            beta_sampled.append(a[0])
            gamma_sampled.append(a[1])
            d_sampled.append(a[2])
            # beta_r= max(0, a[0])
            # gamma_r=max(0, a[1])
            # d_r=max(0, a[2])
            beta_r=a[0]
            gamma_r=a[1]
            d_r=a[2]
            
            deads_sampled=sir_for_optim(np.array([i for i in range(len(self.data) + reach)]), beta_r, gamma_r,d_r)
            prediction_sampled =  deads_sampled[-reach:]
            intervals.append(prediction_sampled)
        self.beta_sampled=beta_sampled
        self.gamma_sampled=gamma_sampled
        self.d_sampled=d_sampled
        intervals=np.array(intervals).transpose()
        self.intervals=intervals
        ci_low=np.array([np.quantile(intervals[i], alpha/2) for i in range(reach)])
        ci_high=np.array([np.quantile(intervals[i],1-alpha/2) for i in range(reach)])
        return prediction, [ci_low, ci_high]
        

    


       
