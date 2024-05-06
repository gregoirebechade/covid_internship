from Model import Model, Multi_Dimensional_Model
from scipy.optimize import curve_fit, minimize
import numpy as np
from scipy.optimize import differential_evolution

import pandas as pd
import sys
sys.path.append('./../')
import scipy.stats
from useful_functions import shift



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
        if len(theta)==2: 
            theta_plus=theta.copy()
            theta_plus[i]+=0.0001
            _, _, _, deads_grad = run_sir([x0[0], x0[1], x0[2], x0[3]], theta_plus[0], 0.2, theta_plus[1], reach+1, 0.001)
            _, _, _, deads = run_sir([x0[0], x0[1], x0[2], x0[3]], theta[0], 0.2, theta[1], reach+1, 0.001)
        elif len(theta)==3: 
            theta_plus=theta.copy()
            theta_plus[i]+=0.0001
            _, _, _, deads_grad = run_sir([x0[0], x0[1], x0[2], x0[3]], theta_plus[0],  theta_plus[1],theta_plus[2],  reach+1, 0.001)
            _, _, _, deads = run_sir([x0[0], x0[1], x0[2], x0[3]], theta[0], theta[1], theta[2],  reach+1, 0.001)
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
    def choose_model(self, gamma_constant, delta_method): 
        self.gamma_constant=gamma_constant
        self.delta_method=delta_method
    def train(self, train_dates, data):
        self.data=data
        self.train_dates=train_dates
        gamma_constant=self.gamma_constant
        if gamma_constant: 
            print('gamma constant')
            p,cov= curve_fit(sir_for_optim_2, self.train_dates,data, p0=[ 5.477e-01  , 5.523e-04],  bounds=([0,0], [np.inf,np.inf]))
            self.beta=p[0]
            self.d=p[1]
            self.gamma=0.2
            self.cov=cov
            self.trained= True
        else : 
            print('gamma not constant ')
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
        self.prediction =  deads[-reach:]
        prediction=self.prediction
        if method == 'covariance': 
            perr = np.sqrt(np.diag(self.cov)) # Idea from: https://github.com/philipgerlee/Predicting-regional-COVID-19-hospital-admissions-in-Sweden-using-mobility-data.
        self.perr=perr
        delta_method=self.delta_method
        if delta_method: 
            print('delta-method')
            ci_low=[]
            ci_high=[]
            if self.gamma_constant: 
                grad=grad_theta_h_theta([self.S[-1], self.I[-1], self.R[-1], self.D[-1]], [self.beta, self.d], reach) # size 2 x reach
            else : 
                grad=grad_theta_h_theta([self.S[-1], self.I[-1], self.R[-1], self.D[-1]], [self.beta,self.gamma,  self.d], reach) # size 3 x reach
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
            intervals=[np.array(prediction)]
            beta_sampled=[]
            gamma_sampled=[]
            d_sampled=[]
            for i in range(100):
                if self.gamma_constant: 
                    a=np.random.multivariate_normal([self.beta,self.d], self.cov, 1)[0] # not sampling along gamma because gamma is centered on zero so when we sample along gamma and we resample when the value of one of the component is over zero, we elminiate half of the values and have very bad predictions
                    while not (a>0).all(): 
                        a=np.random.multivariate_normal([self.beta,self.d], self.cov, 1)[0]
                    beta_sampled.append(a[0])
                    gamma_sampled.append(0.2)
                    d_sampled.append(a[1])
                else: 
                    a=np.random.multivariate_normal([self.beta,self.gamma, self.d], self.cov, 1)[0] # not sampling along gamma because gamma is centered on zero so when we sample along gamma and we resample when the value of one of the component is over zero, we elminiate half of the values and have very bad predictions
                    while not (a>0).all(): 
                        a=np.random.multivariate_normal([self.beta,self.gamma, self.d], self.cov, 1)[0]
                    beta_sampled.append(a[0])
                    gamma_sampled.append(a[1])
                    d_sampled.append(a[2])
                beta_r=beta_sampled[-1]
                d_r=d_sampled[-1]
                gamma_r=gamma_sampled[-1]
                _, _, _, deads_sampled = run_sir([self.S[-1], self.I[-1], self.R[-1], self.D[-1]], beta_r, gamma_r, d_r, reach+1, 0.001)
                d_arr=np.array(differenciate(np.array(deads_sampled)))
                prediction_sampled=d_arr
                intervals.append(prediction_sampled)
            self.beta_sampled=beta_sampled
            self.d_sampled=d_sampled
            self.gamma_sampled=gamma_sampled
            intervals=np.array(intervals).transpose()
            self.intervals=intervals
            ci_low=np.array([np.quantile(intervals[i], alpha/2) for i in range(reach)])
            ci_high=np.array([np.quantile(intervals[i],1-alpha/2) for i in range(reach)])
        return prediction, [ci_low, ci_high]
        




def run_sir_m(x0, a, b , gamma,d, mobility , dt):
    t=len(mobility)
    x=x0
    S=[x[0]]
    I=[x[1]]
    R=[x[2]]
    D=[x[3]] # deads 
    n_iter=int(t/dt)
    N=sum(x0)
    for i in range(n_iter):
        todays_mobility=mobility[int(i*dt)]
        beta=a*todays_mobility+b
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

def sir_for_optim_m( x, a, b ,d, mobility): # returns first the number of deaths and then the number of total infected
    
    s_0=1000000 -1
    i_0=1
    r_0=0
    d_0=0
    x0=np.array([s_0, i_0, r_0, d_0])
    dt=0.001

    S, I, R, D = run_sir_m(x0, a, b , 0.2,d, mobility ,   dt)
    zer=np.array([0])
    d_arr=np.array(D)
    I_arr=np.array(I)
    return np.concatenate((differenciate(np.concatenate((zer,d_arr))), I_arr))




def sir_for_optim_normalized(x, a, b, d, mobility, new_deaths, n_infected, shift1= 0, shift2 = 0 , taking_I_into_account=True): # returns firts the number of deaths and then the number of total infected
    I_and_D=sir_for_optim_m(x, a, b, d, mobility)
    I=I_and_D[len(I_and_D)//2:]
    D=I_and_D[:len(I_and_D)//2]
    if taking_I_into_account: 
        return np.concatenate((shift(D, shift1)/np.max(new_deaths), shift(I, shift2)/np.max(n_infected)))
    else:
        return D




def grad_theta_h_theta_m(x0, theta, mob_predicted ): 
    reach=len(mob_predicted) 
    grad=np.zeros((len(theta), reach))
    for i in range(len(grad)): 
        theta_plus=theta.copy()
        theta_plus[i]+=0.0001
        mob_extended=np.concatenate((mob_predicted, np.array([mob_predicted[-1]])))
        _, _, _, deads_grad = run_sir_m([x0[0], x0[1], x0[2], x0[3]], theta_plus[0], theta_plus[1], 0.2,theta_plus[2] , mob_extended, 0.001)
        _, _, _, deads= run_sir_m([x0[0], x0[1], x0[2], x0[3]], theta[0], theta[1], 0.2, theta[2], mob_extended, 0.001)
        d_arr_grad= np.array(differenciate(np.array(deads_grad)))
        d_arr=np.array(differenciate(np.array(deads)))
        grad[i]=(d_arr_grad-d_arr)/0.0001
    return grad

class Multi_SIRD_model(Multi_Dimensional_Model): 
    s_0=1000000 -1
    i_0=1
    r_0=0
    d_0=0
    dt=0.001
    def choose_model(self, taking_I_into_account, shifts):
        if taking_I_into_account: 
            print('Taking I into account')
        else : 
            print('Not taking I into account')
        if shifts: 
            print('shifting')
        else: 
            print('not shifting')
        self.taking_I_into_account=taking_I_into_account
        self.shifts=shifts

    def train(self, train_dates, data):
        self.data=data
        self.train_dates=train_dates
        # curve = lambda x, a, b, d, n :  sir_for_optim_normalized(x, a, b, d, shift(data[2], n), data[0], data[1], taking_I_into_account) 
        # curve = lambda x, a, b, d, n : (n-int(n))*  sir_for_optim_normalized(x, a, b, d, shift(data[2], int(n)), data[0], data[1], taking_I_into_account) + (1-(n-int(n))) * sir_for_optim_normalized(x, a, b, d, shift(data[2], int(n)+1), data[0], data[1], taking_I_into_account)
        if self.taking_I_into_account: 
            obj=np.concatenate((np.array(data[0])/max(np.array(data[0])), np.array(data[1])/max(np.array(data[1]))))
            coef=2
        else: 
            obj=np.array(data[0]/max(np.array(data[0])))
            coef=1
        if self.shifts: 
            if not self.taking_I_into_account:
                print("It doesn't make sense to take the shifts into account if we do not take I into account")
            # p,cov= curve_fit(curve2,np.array([i for i in range(coef*len(train_dates))]),obj, p0=[ 1, 1 , 5.523e-04, 5, 10],  bounds=([-np.inf, -np.inf, 0, -np.inf, -np.inf], [np.inf,np.inf, np.inf, np.inf, np.inf]))
            method1=False
            if method1: 
                print('direct optimization')
                x=np.array([i for i in range(coef*len(train_dates))])
                curve2 = lambda params :     ((1- (params[3] - int(params[3]))) *np.sum(( sir_for_optim_normalized(x, params[0], params[1], params[2], self.data[1], self.data[0], self.data[1], shift1=int(params[3]), shift2= int(params[4]), taking_I_into_account=self.taking_I_into_account) - obj )**2)
                                            + ((params[3] - int(params[3]))) *np.sum(( sir_for_optim_normalized(x, params[0], params[1], params[2], self.data[1], self.data[0], self.data[1], shift1=int(params[3])+1, shift2= int(params[4]), taking_I_into_account=self.taking_I_into_account) - obj )**2)
                                            + (1-(params[4] - int(params[4]))) *np.sum(( sir_for_optim_normalized(x, params[0], params[1], params[2], self.data[1], self.data[0], self.data[1], shift1=int(params[3]), shift2= int(params[4]), taking_I_into_account=self.taking_I_into_account) - obj )**2)
                                            + ((params[4] - int(params[4]))) *np.sum(( sir_for_optim_normalized(x, params[0], params[1], params[2], self.data[2], self.data[0], self.data[1], shift1=int(params[3]), shift2= int(params[4])+1, taking_I_into_account=self.taking_I_into_account)- obj )**2))
                res=minimize(curve2, [1, 1, 5.523e-04, 5, 10],  bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (0, np.inf), (-np.inf, np.inf), (-np.inf, np.inf)])
                p=res.x
                self.a=p[0]
                self.b=p[1]
                self.d=p[2]
                self.shift1=p[3]
                self.shift2=p[4]
                self.cov=res.hess_inv            
            else: 
                print(' grid search on the shifts')
                dico_losses=dict()
                best_result_so_far=np.inf
                best_p=None
                best_cov=None
                best_shift1=None
                best_shift2=None
                for shift1 in range(1): 
                    for shift2 in range(-15, 0):
                        print(shift1, shift2)
                        curve1 = lambda x, a, b, d :   sir_for_optim_normalized(x, a, b, d, self.data[2], data[0], data[1], shift1=shift1, shift2= shift2, taking_I_into_account=self.taking_I_into_account)
                        try: 
                            p, cov= curve_fit(curve1,np.array([i for i in range(coef*len(train_dates))]),obj, p0=[ 1, 1 , 5.523e-04],  bounds=([-np.inf, -np.inf, 0], [np.inf,np.inf, np.inf]))
                            local_result=np.sum((curve1(np.array([i for i in range(coef*len(train_dates))]), p[0], p[1], p[2])-obj)**2)
                            dico_losses[str(shift1) + ' ' + str(shift2)]=local_result
                        except RuntimeError: 
                            print('oups')
                            dico_losses[str(shift1) + ' ' + str(shift2)]=np.inf
                        if local_result<best_result_so_far:
                            print('new best result !!')
                            print('a = ', p[0])
                            print(' b = ', p[1])
                            print('d = ', p[2] )
                            print(' shift1 = ', shift1)
                            print('shift2 = ', shift2)
                            print('and the local result is..... ', local_result)
                            print()
                            print()
                            best_result_so_far=local_result
                            best_p=p
                            best_cov=cov
                            best_shift1=shift1
                            best_shift2=shift2
                self.shift1=best_shift1
                self.shift2=best_shift2
                self.p=best_p
                self.a=self.p[0]
                self.b=self.p[1]
                self.d=self.p[2]
                self.cov=best_cov
                self.all_losses=dico_losses
        else:
            curve1 = lambda x, a, b, d :   sir_for_optim_normalized(x, a, b, d, self.data[2], data[0], data[1], shift1=0, shift2= 0, taking_I_into_account=self.taking_I_into_account)
            p,cov= curve_fit(curve1,np.array([i for i in range(coef*len(train_dates))]),obj, p0=[ 1, 1 , 5.523e-04],  bounds=([-np.inf, -np.inf, 0], [np.inf,np.inf, np.inf]))
            self.a=p[0]
            self.b=p[1]
            self.d=p[2]
            self.cov=cov
        self.gamma=0.2
        self.trained= True
    def predict(self, reach,  alpha, method='covariance'):
        mob_predicted=np.array([self.data[2][-1] for i in range(reach)])
        reach=len(mob_predicted)
        s_0=1000000 -1
        i_0=1
        r_0=0
        d_0=0
        S,I,R,D=run_sir_m([s_0, i_0, r_0, d_0], self.a, self.b,0.2,  self.d ,self.data[2], 0.001)
        self.S=S
        self.I=I
        self.R=R
        self.D=D
        assert self.trained, 'The model has not been trained yet'
        deads_and_n_infected=sir_for_optim_m(None, self.a, self.b,self.d, np.concatenate((np.array(self.data[2]), mob_predicted)))
        deads=deads_and_n_infected[:len(np.array(self.data[2]))+len(mob_predicted)]
        self.prediction =  deads[-reach:]
        prediction=self.prediction
        if method == 'covariance': 
            perr = np.sqrt(np.diag(self.cov)) # Idea from: https://github.com/philipgerlee/Predicting-regional-COVID-19-hospital-admissions-in-Sweden-using-mobility-data.
        self.perr=perr
        delta_method=True
        if delta_method: 
            ci_low=[]
            ci_high=[]
            mob_extended=np.concatenate( ((np.array([mob_predicted[0]]), mob_predicted)))
            grad=grad_theta_h_theta_m([self.S[-1], self.I[-1], self.R[-1], self.D[-1]], [self.a, self.b , self.d], mob_predicted) # size 3 x reach
            cov=self.cov
            vars=np.diagonal((grad.transpose() @ cov @ grad).transpose())
           
            assert(len(vars)==reach), str(len(vars) + 'different from ' + str(reach))
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



