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
    t=len(x-1)
    S,I,R,D=run_sir(x0, beta, gamma,d,  t, dt)
    zer=np.array([0])
    d_arr=np.array(D)
    return differenciate(np.concatenate((zer,d_arr))) # returns a value per day

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


    def predict(self, reach, alpha):
        assert self.trained, 'The model has not been trained yet'
        deads=sir_for_optim(np.array([i for i in range(len(self.train_dates)+reach)]), self.beta, self.gamma,self.d)
        prediction =  deads[-reach:]
        perr = np.sqrt(np.diag(self.cov)) # Idea from: https://github.com/philipgerlee/Predicting-regional-COVID-19-hospital-admissions-in-Sweden-using-mobility-data.
        # we only take into account the approximation error and not the variance of the new_deaths. 
        intervals=[prediction]
        beta_sampled=[]
        gamma_sampled=[]
        d_sampled=[]
        for i in range(100): 
            beta_r=max(0, np.random.normal(self.beta, perr[0], 1)[0])
            gamma_r=max(0,np.random.normal(self.gamma, perr[1], 1)[0])
            d_r=max(0,np.random.normal(self.d, perr[2], 1)[0])
            beta_sampled.append(beta_r)
            gamma_sampled.append(gamma_r)
            d_sampled.append(d_r)
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
        

    


       
