from Model import Model
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import sys
sys.path.append('./../')
import scipy.stats



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


def get_sousmatrice(matrix): 
    result=np.zeros((2,2))

    result[0][0]=matrix[0][0]
    result[0][1]=matrix[0][2]
    result[1][0]=matrix[2][0]
    result[1][1]=matrix[2][2]
    return result


class SIRD_model(Model): 
    s_0=1000000 -1
    i_0=1
    r_0=0
    d_0=0
    dt=0.001
    def train(self, train_dates, data):
        self.data=data
        self.train_dates=train_dates
        # p,cov= curve_fit(sir_for_optim, self.train_dates,data, p0=[ 4.37, 2.2, 2.0],  bounds=([0,0,0], [5,5,5]))
        p,cov= curve_fit(sir_for_optim, self.train_dates,data, p0=[ 5.477e-01 , 2.555e-02 , 5.523e-04],  bounds=([0,0,0], [10,5,5]))
        self.beta=p[0]
        self.gamma=p[1]
        self.d=p[2]
        self.cov=cov
        self.trained= True


    def predict(self, reach, alpha, method='covariance'):
        S,I,R,D=run_sir([s_0, i_0, r_0, d_0], self.beta, self.gamma, self.d , len(self.train_dates), 0.001)
        self.S=S
        self.I=I
        self.R=R
        self.D=D
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
            self.cov=cov
            perr=np.sqrt(abs(np.diag(cov)))
            self.perr=perr
        elif method == 'delta': 
            print('delta')
            p=[self.beta, self.gamma, self.d]
            sigma2=estimate_sigma2(self.data, deads[:-reach], len(p)) * np.identity(len(self.data))
            grad=grad_f_for_delta_method(self.train_dates, self.data)
            perr = np.sqrt(np.diag(np.matmul(np.matmul(grad.transpose(), sigma2) , grad)))
            self.perr = perr

        intervals=[np.array(prediction)]
        beta_sampled=[]
        gamma_sampled=[]
        d_sampled=[]
        cov_2=get_sousmatrice(self.cov)

        for i in range(100): 
            # beta_r=max(0, np.random.normal(self.beta, perr[0], 1)[0])
            # gamma_r=max(0,np.random.normal(self.gamma, perr[1], 1)[0])
            # d_r=max(0,np.random.normal(self.d, perr[2], 1)[0])
            # a=np.random.multivariate_normal([self.beta,self.gamma,self.d], self.cov, 1)[0]

            
            a=np.random.multivariate_normal([self.beta,self.d], cov_2, 1)[0] # not sampling along gamma because gamma is centered on zero so when we sample along gamma and we resample when the value of one of the component is over zero, we elminiate half of the values and have very bad predictions
            while not (a>0).all(): 
                a=np.random.multivariate_normal([self.beta,self.d], cov_2, 1)[0]
            
            # a[1]=abs(a[1])
            # while not (a>0).all(): 
            #     a=np.random.multivariate_normal([self.beta,self.gamma,self.d], self.cov, 1)[0]
                # a[1]=abs(a[1])
            beta_sampled.append(a[0])
            gamma_sampled.append(self.gamma)
            d_sampled.append(a[1])
            # beta_r= max(0, a[0])
            # gamma_r=max(0, a[1])
            # d_r=max(0, a[2])
            beta_r=a[0]
            # gamma_r=a[1]
            gamma_r = self.gamma
            d_r=a[1]
            

            # sampling with sir for optim with startpoint = 0
            # deads_sampled=sir_for_optim(np.array([i for i in range(len(self.data) + reach)]), beta_r, gamma_r,d_r)
            # prediction_sampled =  deads_sampled[-reach:]
 
            # sampling with run_sir  with startpoint = last datapoint: 
            s_sampled, i_sampled, r_sampled, deads_sampled = run_sir([self.S[-1], self.I[-1], self.R[-1], self.D[-1]], beta_r, gamma_r, d_r, reach+1, 0.001)
            zer=np.array([D[-1]])
            last_new_deaths=np.array([self.D[-1]- self.D[-2]])
            d_arr=np.array(differenciate(np.array(deads_sampled)))
            # print('last new deaths:')
            # print((last_new_deaths))
            # print('darr: ')
           
            # print((d_arr))
            prediction_sampled= (np.concatenate((last_new_deaths,d_arr))) # returns a value per day
            prediction_sampled=d_arr
            # print('prediction sampled: ')
            # print(prediction_sampled)
            # print()
            # print('----------------')

            intervals.append(prediction_sampled)

        
        self.beta_sampled=beta_sampled
        self.gamma_sampled=gamma_sampled
        self.d_sampled=d_sampled
        intervals=np.array(intervals).transpose()
        self.intervals=intervals
        ci_low=np.array([np.quantile(intervals[i], alpha/2) for i in range(reach)])
        ci_high=np.array([np.quantile(intervals[i],1-alpha/2) for i in range(reach)])
        return prediction, [ci_low, ci_high]
        




def sir_for_optim_2(x, beta, d):
    # x is a list of dates (0 - 122)
    x0=[s_0, i_0, r_0, d_0]
    t=len(x)
    S,I,R,D=run_sir(x0, beta, 0.2,d,  t, dt)
    zer=np.array([0])
    d_arr=np.array(D)
    return differenciate(np.concatenate((zer,d_arr))) # returns a value per day


def grad_theta_h_theta(x0, theta, reach ): 
    grad=np.zeros((len(theta), reach))
    for i in range(len(grad)): 
        theta_plus=theta.copy()
        theta_plus[i]+=0.0001
        _, _, _, deads_grad = run_sir([x0[0], x0[1], x0[2], x0[3]], theta_plus[0], 0.2, theta_plus[1], reach+1, 0.001)
        _, _, _, deads = run_sir([x0[0], x0[1], x0[2], x0[3]], theta[0], 0.2, theta[1], reach+1, 0.001)
        d_arr_grad=np.array(differenciate(np.array(deads_grad)))
        d_arr=np.array(differenciate(np.array(deads)))
        grad[i]=(d_arr_grad-d_arr)/0.0001
    return grad



class SIRD_model_2(Model): 
    s_0=1000000 -1
    i_0=1
    r_0=0
    d_0=0
    dt=0.001
    def train(self, train_dates, data):
        self.data=data
        self.train_dates=train_dates
        p,cov= curve_fit(sir_for_optim_2, self.train_dates,data, p0=[ 5.477e-01  , 5.523e-04],  bounds=([0,0], [np.inf,np.inf]))
        self.beta=p[0]
        self.d=p[1]
        self.gamma=0.2
        self.cov=cov
        self.trained= True
    def predict(self, reach, alpha, method='covariance'):
        S,I,R,D=run_sir([s_0, i_0, r_0, d_0], self.beta, self.gamma, self.d , len(self.train_dates), 0.001)
        self.S=S
        self.I=I
        self.R=R
        self.D=D
        assert self.trained, 'The model has not been trained yet'
        deads=sir_for_optim(np.array([i for i in range(len(self.train_dates)+reach)]), self.beta, self.gamma,self.d)
        self.prediction =  deads[-reach:]
        prediction=self.prediction
        if method == 'covariance': 
            perr = np.sqrt(np.diag(self.cov)) # Idea from: https://github.com/philipgerlee/Predicting-regional-COVID-19-hospital-admissions-in-Sweden-using-mobility-data.
        self.perr=perr
        intervals=[np.array(prediction)]
        beta_sampled=[]
        d_sampled=[]
        for i in range(100):
            a=np.random.multivariate_normal([self.beta,self.d], self.cov, 1)[0] # not sampling along gamma because gamma is centered on zero so when we sample along gamma and we resample when the value of one of the component is over zero, we elminiate half of the values and have very bad predictions
            while not (a>0).all(): 
                a=np.random.multivariate_normal([self.beta,self.d], self.cov, 1)[0]
            beta_sampled.append(a[0])
            d_sampled.append(a[1])
            beta_r=a[0]
            d_r=a[1]
            _, _, _, deads_sampled = run_sir([self.S[-1], self.I[-1], self.R[-1], self.D[-1]], beta_r, self.gamma, d_r, reach+1, 0.001)
            d_arr=np.array(differenciate(np.array(deads_sampled)))
            prediction_sampled=d_arr
            intervals.append(prediction_sampled)
        self.beta_sampled=beta_sampled
        self.d_sampled=d_sampled
        intervals=np.array(intervals).transpose()
        self.intervals=intervals
        ci_low=np.array([np.quantile(intervals[i], alpha/2) for i in range(reach)])
        ci_high=np.array([np.quantile(intervals[i],1-alpha/2) for i in range(reach)])
        delta_method=True
        if delta_method: 
            print('delta-method')
            ci_low=[]
            ci_high=[]
            grad=grad_theta_h_theta([self.S[-1], self.I[-1], self.R[-1], self.D[-1]], [self.beta, self.d], reach) # size 3 x reach
            cov=self.cov
            vars=np.diagonal((grad.transpose() @ cov @ grad).transpose())
            print(vars)
            assert(len(vars)==reach, str(len(vars)) + 'different from ' + str(reach))
            for i in range(len(vars)): 
                down = scipy.stats.norm.ppf(alpha/2, loc=self.prediction[i], scale=np.sqrt(vars[i]))
                ci_low.append(down)
                up = scipy.stats.norm.ppf(1-(alpha/2), loc=self.prediction[i], scale=np.sqrt(vars[i]))
                ci_high.append(up)
            self.ci_low=ci_low
            self.ci_high=ci_high
        else: 
            print('sampling parameters')
        return prediction, [ci_low, ci_high]
        

def sir_reset_state(x, beta, gamma, d, s0, i0, r0, d0 ): 
    _,_,_,D=  run_sir([s0, i0, r0, d0], beta, gamma, d , len(x), 0.001)
    zer=np.array([0])
    d_arr=np.array(D)
    return differenciate(np.concatenate((zer,d_arr))) 

       
class SIRD_model_15_days(Model): 
    
    def train(self, train_dates, data):
        self.data=data
        self.train_dates=train_dates
        p,cov= curve_fit(sir_reset_state, self.train_dates[-15:],data[-15:], p0=[ 5.477e-01 , 2.555e-02 , 5.523e-04, 1000000 - data[-15]*11, 10 * data[-15], 1, data[-15]],  bounds=([0,0,0, 0, 0, 0, 0], [5,5,5, np.inf, np.inf, np.inf, np.inf]))
        self.beta=p[0]
        self.gamma=p[1]
        self.d=p[2]
        self.s0=p[3]
        self.i0=p[4]
        self.r0=p[5]
        self.d0=p[6]
        self.cov=cov
        self.trained= True


    def predict(self, reach, alpha, method='covariance'):
        assert self.trained, 'The model has not been trained yet'
        deads=sir_reset_state(np.array([i for i in range(15 + reach)]), self.beta, self.gamma,self.d, self.s0, self.i0, self.r0, self.d0)
        prediction =  deads[-reach:]
        if method == 'covariance': 
            perr = np.sqrt(np.diag(self.cov)) # Idea from: https://github.com/philipgerlee/Predicting-regional-COVID-19-hospital-admissions-in-Sweden-using-mobility-data.
        
        intervals=[prediction]
        beta_sampled=[]
        gamma_sampled=[]
        d_sampled=[]
        s0_sampled=[]
        i0_sampled=[]
        r0_sampled=[]
        d0_sampled=[]

        for i in range(100): 
            a=np.random.multivariate_normal([self.beta,self.gamma, self.d, self.s0, self.i0, self.r0, self.d0 ], self.cov, 1)[0] 
            while not (a>0).all(): 
                a=np.random.multivariate_normal([self.beta,self.gamma, self.d, self.s0, self.i0, self.r0, self.d0 ], self.cov, 1)[0] 
            
            
            beta_sampled.append(a[0])
            gamma_sampled.append(self.gamma)
            d_sampled.append(a[1])
            s0_sampled.append(a[2])
            i0_sampled.append(a[3])
            r0_sampled.append(a[4])
            d0_sampled.append(a[5])
            beta_r=a[0]
            gamma_r = a[1]
            d_r=a[2]
            s0_r=a[3]
            i0_r=a[4]
            r0_r=a[5]
            d0_r=a[6]
            deads_sampled=sir_reset_state(np.array([i for i in range(15 + reach)]), beta_r, gamma_r,d_r, s0_r, i0_r, r0_r, d0_r)
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
        

