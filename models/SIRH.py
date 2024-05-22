from Model import Model, Multi_Dimensional_Model
from scipy.optimize import curve_fit, minimize
import numpy as np
from scipy.optimize import differential_evolution

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
h_0=0
t=len(death_cumul-1)
dt=0.001


def derive_sirh(x, beta, N, gamma_i,gamma_h,  h):
    S=x[0]
    I=x[1]
    R=x[2]
    H=x[3]
    return np.array([-beta*S*I/N, beta*S*I/N - (gamma_i+h)*I  , gamma_i*I + gamma_h *H, h * I - gamma_h * H ])



def run_sirh(x0, beta, gamma_i, gamma_h,h,  t, dt):
    
    x=x0
    S=[x[0]]
    I=[x[1]]
    R=[x[2]]
    H=[x[3]] # hospitalized 
    n_iter=int(t/dt)
    N=sum(x0)
    for i in range(n_iter):
        x=x+dt*derive_sirh(x, beta, N, gamma_i, gamma_h, h)
        S.append(x[0])
        I.append(x[1])
        R.append(x[2])
        H.append(x[3])
    s_final=[]
    i_final=[]
    r_final=[]
    h_final=[]
    time=np.linspace(0, t, int(t/dt) )
    for i in range(len(time)-1):
        if abs(time[i]-int(time[i]))<dt: 
            s_final.append(S[i])
            i_final.append(I[i])
            r_final.append(R[i])
            h_final.append(H[i])
    return s_final, i_final, r_final, h_final
    



def differenciate(x): 
    dx=[x[i+1]-x[i] for i in range(len(x)-1)]
    return dx

def sirh_for_optim(x, beta, gamma_i, gamma_h, h):
    # x is a list of dates (0 - 122)
    x0=[s_0, i_0, r_0, h_0]
    t=len(x)
    S,I,R,H=run_sirh(x0, beta, gamma_i, gamma_h,h,  t, dt)
    h_arr=np.array(H)
    return h_arr 





def sirh_for_optim_3(x, beta,gamma_h,  h):
    # x is a list of dates (0 - 122)
    x0=[s_0, i_0, r_0, h_0]
    t=len(x)
    S,I,R,H=run_sirh(x0, beta, 0.2,gamma_h, h,  t, dt)
    h_arr=np.array(H)
    return h_arr 


def sirh_for_optim_4(x, beta, gamma_i,  h):
    # x is a list of dates (0 - 122)
    x0=[s_0, i_0, r_0, h_0]
    t=len(x)
    S,I,R,H=run_sirh(x0, beta, gamma_i, 0.2, h,  t, dt)
    h_arr=np.array(H)
    return h_arr

def sirh_for_optim_2(x, beta, h):
    # x is a list of dates (0 - 122)
    x0=[s_0, i_0, r_0, h_0]
    t=len(x)
    S,I,R,H=run_sirh(x0, beta, 0.2, 0.2, h,  t, dt)
    h_arr=np.array(H)
    return h_arr 

def grad_theta_h_theta(x0, theta, reach, the_gamma_constant=None): 
    grad=np.zeros((len(theta), reach))
    for i in range(len(grad)): 
        if len(theta)==2: 
            theta_plus=theta.copy()
            theta_plus[i]+=0.0001
            _, _, _, hospitalized_grad = run_sirh([x0[0], x0[1], x0[2], x0[3]], theta_plus[0], 0.2,0.2,  theta_plus[1], reach, 0.001)
            _, _, _, hospitalized = run_sirh([x0[0], x0[1], x0[2], x0[3]], theta[0], 0.2, 0.2, theta[1], reach, 0.001)
        elif len(theta)==4: 
            theta_plus=theta.copy()
            theta_plus[i]+=0.0001
            _, _, _, hospitalized_grad = run_sirh([x0[0], x0[1], x0[2], x0[3]], theta_plus[0],  theta_plus[1],theta_plus[2], theta_plus[3],  reach, 0.001)
            _, _, _, hospitalized = run_sirh([x0[0], x0[1], x0[2], x0[3]], theta[0], theta[1], theta[2],theta_plus[3],  reach, 0.001)
        elif len(theta)==3: 
            theta_plus=theta.copy()
            theta_plus[i]+=0.0001
            if the_gamma_constant=='gamma_h': 
                _, _, _, hospitalized_grad = run_sirh([x0[0], x0[1], x0[2], x0[3]], theta_plus[0],  theta_plus[1],0.2,  theta_plus[2],  reach, 0.001)
                _, _, _, hospitalized = run_sirh([x0[0], x0[1], x0[2], x0[3]], theta[0], theta[1], 0.2, theta[2],  reach, 0.001)
            elif the_gamma_constant=='gamma_i': 
                _, _, _, hospitalized_grad = run_sirh([x0[0], x0[1], x0[2], x0[3]], theta_plus[0],  0.2,theta_plus[1],  theta_plus[2],  reach, 0.001)
                _, _, _, hospitalized = run_sirh([x0[0], x0[1], x0[2], x0[3]], theta[0],  0.2, theta[1], theta[2],  reach, 0.001)


        h_arr_grad=(np.array(hospitalized_grad))
        h_arr=(np.array(hospitalized))
        grad[i]=(h_arr_grad-h_arr)/0.0001
    return grad



class SIRH_model_2(Model): 
    s_0=1000000 -1
    i_0=1
    r_0=0
    h_0=0
    dt=0.001
    def choose_model(self, gamma_i_constant,gamma_h_constant,  reset_state): 
        self.gamma_i_constant=gamma_i_constant
        self.gamma_h_constant=gamma_h_constant
        self.reset_state=reset_state
        self.N=s_0+i_0+r_0+h_0
    def train( self, train_dates,  data):
        self.name='SIRH'
        self.data=data
        self.train_dates=train_dates
        if self.gamma_i_constant and self.gamma_h_constant: 
            # print('gamma_i and gamma_h constants')
            p,cov= curve_fit(sirh_for_optim_2, [i for i in range(len(data))],data, p0=[ 5.477e-01  , 5.523e-04],  bounds=([0,0], [np.inf,np.inf]))
            self.beta=p[0]
            self.h=p[1]
            self.gamma_i=0.2
            self.gamma_h=0.2
            self.cov=cov
            self.trained= True
        elif not self.gamma_i_constant and not self.gamma_h_constant: 
            # print('gamma_i not constant and gamma_h not constant ')
            p,cov= curve_fit(sirh_for_optim, self.train_dates,data, p0=[ 5.477e-01 , 2.555e-02 ,2.555e-02,  5.523e-04],  bounds=([0,0,0, 0], [10,10, 10, 10]))
            self.beta=p[0]
            self.gamma_i=p[1]
            self.gamma_h=p[2]
            self.h=p[3]
            self.cov=cov
            self.trained= True
        elif self.gamma_i_constant and not self.gamma_h_constant: 
            # print('on fixe gamma_i mais pas gamma_h')
            p,cov= curve_fit(sirh_for_optim_3, self.train_dates,data, p0=[ 5.477e-01 , 2.555e-02 ,  5.523e-04],  bounds=([0,0, 0], [10,10, 10]))
            self.beta=p[0]
            self.gamma_i=0.2
            self.gamma_h=p[1]
            self.h=p[2]
            self.cov=cov
            self.trained= True
        else: 
            # print('on fixe gamma_h mais pas gamma_i')
            p,cov= curve_fit(sirh_for_optim_4, self.train_dates,data, p0=[ 5.477e-01 , 2.555e-02 ,  5.523e-04],  bounds=([0,0, 0], [10,10, 10]))
            self.beta=p[0]
            self.gamma_i=p[1]
            self.gamma_h=0.2
            self.h=p[2]
            self.cov=cov
            self.trained= True

    def predict(self, reach, alpha):
        S,I,R,H=run_sirh([s_0, i_0, r_0, h_0], self.beta, self.gamma_i, self.gamma_h, self.h , len(self.train_dates), 0.001)
        self.S=S
        self.I=I
        self.R=R
        self.H=H
        assert self.trained, 'The model has not been trained yet'
        if self.reset_state: 
            s_t=S[-1]
            i_t=I[-1]
            h_t=self.data[-1]
            r_t=self.N-s_t-i_t-h_t
            _, _, _, hospitalized=run_sirh([s_t, i_t, r_t, h_t], self.beta, self.gamma_i, self.gamma_h, self.h, reach, 0.001)
        else : 
            hospitalized=sirh_for_optim(np.array([i for i in range(len(self.train_dates)+reach)]), self.beta, self.gamma_i, self.beta_h,self.h)
        self.prediction =  hospitalized
        prediction=self.prediction
        if True: 
            # print('delta-method')
            ci_low=[]
            ci_high=[]
            if self.gamma_i_constant and self.gamma_h_constant: 
                grad=grad_theta_h_theta([self.S[-1], self.I[-1], self.R[-1], self.H[-1]], [self.beta, self.h], reach) # size 2 x reach
            elif not self.gamma_i_constant and not self.gamma_h_constant: 
                grad=grad_theta_h_theta([self.S[-1], self.I[-1], self.R[-1], self.H[-1]], [self.beta,self.gamma_i,self.gamma_h,   self.h], reach) # size 4 x reach
            elif self.gamma_i_constant and not self.gamma_h_constant:
                grad=grad_theta_h_theta([self.S[-1], self.I[-1], self.R[-1], self.H[-1]], [self.beta,self.gamma_h,  self.h], reach, the_gamma_constant='gamma_i')
            else:
                grad=grad_theta_h_theta([self.S[-1], self.I[-1], self.R[-1], self.H[-1]], [self.beta,self.gamma_i,  self.h], reach, the_gamma_constant='gamma_h')
            cov=self.cov
            vars=np.diagonal((grad.transpose() @ cov @ grad).transpose())
            assert(len(vars)==reach, str(len(vars)) + 'different from ' + str(reach))
            for i in range(len(vars)): 
                down = scipy.stats.norm.ppf(alpha/2, loc=self.prediction[i], scale=np.sqrt(vars[i]))
                ci_low.append(down)
                up = scipy.stats.norm.ppf(1-(alpha/2), loc=self.prediction[i], scale=np.sqrt(vars[i]))
                ci_high.append(up)
            self.ci_low=ci_low
            self.ci_high=ci_high
        return prediction, [ci_low, ci_high]
        




def run_sirh_m(x0, a, b , gamma_i, gamma_h,h, mobility , dt):
    t=len(mobility)
    x=x0
    S=[x[0]]
    I=[x[1]]
    R=[x[2]]
    H=[x[3]] # hospitalized 
    n_iter=int(t/dt)
    N=sum(x0)
    for i in range(n_iter):
        todays_mobility=mobility[int(i*dt)]
        beta=a*todays_mobility+b
        x=x+dt*derive_sirh(x, beta, N, gamma_i, gamma_h, h)
        S.append(x[0])
        I.append(x[1])
        R.append(x[2])
        H.append(x[3])
    s_final=[]
    i_final=[]
    r_final=[]
    h_final=[]
    time=np.linspace(0, t, int(t/dt) )
    for i in range(len(time)-1):
        if abs(time[i]-int(time[i]))<dt: 
            s_final.append(S[i])
            i_final.append(I[i])
            r_final.append(R[i])
            h_final.append(H[i])
    return s_final, i_final, r_final, h_final

def sirh_for_optim_m( x, a, b ,h, mobility, gamma_i=0.2, gamma_h=0.2): # returns first the number of deaths and then the number of total infected
    
    s_0=1000000 -1
    i_0=1
    r_0=0
    h_0=0
    x0=np.array([s_0, i_0, r_0, h_0])
    dt=0.001

    S, I, R, H = run_sirh_m(x0, a, b , gamma_i, gamma_h, h, mobility ,   dt)
    zer=np.array([0])
    h_arr=np.array(H)
    I_arr=np.array(I)
    return np.concatenate((h_arr, I_arr))




def sirh_for_optim_normalized(x, a, b, h, mobility, n_hospitalized, n_infected,  taking_I_into_account=True, gamma_i=0.2, gamma_h=0.2): # returns firts the number of deaths and then the number of total infected
    I_and_H=sirh_for_optim_m(x, a, b, h, mobility, gamma_i, gamma_h)
    I=I_and_H[len(I_and_H)//2:]
    H=I_and_H[:len(I_and_H)//2]
    if taking_I_into_account: 
        return np.concatenate((H/np.max(n_hospitalized), I/np.max(n_infected)))
    else:
        return H



def grad_theta_h_theta_m(x0, theta, mob_predicted ): # for gamma constant
    reach=len(mob_predicted) 
    grad=np.zeros((len(theta), reach))
    for i in range(len(grad)): 
        theta_plus=theta.copy()
        theta_plus[i]+=0.0001
        mob_extended=np.concatenate((mob_predicted, np.array([mob_predicted[-1]])))
        _, _, _, hospitalized_grad = run_sirh_m([x0[0], x0[1], x0[2], x0[3]], theta_plus[0], theta_plus[1], 0.2,0.2, theta_plus[2] , mob_predicted, 0.001)
        _, _, _, hospitalized= run_sirh_m([x0[0], x0[1], x0[2], x0[3]], theta[0], theta[1], 0.2,0.2,  theta[2], mob_predicted, 0.001)
        hospitalized_arr_grad= (np.array(hospitalized_grad))
        hospitalized_arr=(np.array(hospitalized))
        grad[i]=(hospitalized_arr_grad-hospitalized_arr)/0.0001
    return grad




def grad_theta_h_theta_m_bis(x0, theta, mob_predicted ): # for gamma not constants
    reach=len(mob_predicted) 
    grad=np.zeros((len(theta), reach))
    for i in range(len(grad)): 
        theta_plus=theta.copy()
        theta_plus[i]+=0.0001
        _, _, _, hospitalized_grad = run_sirh_m([x0[0], x0[1], x0[2], x0[3]], theta_plus[0], theta_plus[1],  theta_plus[2],  theta_plus[3],  theta_plus[4] , mob_predicted, 0.001)
        _, _, _, hospitalized= run_sirh_m([x0[0], x0[1], x0[2], x0[3]], theta[0], theta[1],  theta[2],theta[3], theta[4],  mob_predicted, 0.001)
        hospitalized_arr_grad= (np.array(hospitalized_grad))
        hospitalized_arr=(np.array(hospitalized))
        grad[i]=(hospitalized_arr_grad-hospitalized_arr)/0.0001
    return grad


class Multi_SIRH_model(Multi_Dimensional_Model): 
    s_0=1000000 -1
    i_0=1
    r_0=0
    h_0=0
    dt=0.001
    def choose_model(self, taking_I_into_account, gamma_constants):
        self.gamma_constants=gamma_constants
        # if taking_I_into_account: 
        #     print('Taking I into account')
        # else : 
        #     print('Not taking I into account')
            
        self.taking_I_into_account=taking_I_into_account
    def train( self,train_dates,   data):
        self.data=data
        self.train_dates=[i for i in range(len(data[0]))]
        if self.taking_I_into_account: 
            obj=np.concatenate((np.array(data[0])/max(np.array(data[0])), np.array(data[1])/max(np.array(data[1]))))
            coef=2
        else: 
            obj=np.array(data[0]/max(np.array(data[0])))
            coef=1
        
        gamma_constants=self.gamma_constants
        if gamma_constants: 
            self.gamma_constants=True
            curve1 = lambda x, a, b, h :   sirh_for_optim_normalized(x, a, b, h, self.data[2], data[0], data[1], taking_I_into_account=self.taking_I_into_account)
            p,cov= curve_fit(curve1,np.array([i for i in range(coef*len(self.train_dates))]),obj, p0=[ 1, 1 , 5.523e-04],  bounds=([-np.inf, -np.inf, 0], [np.inf,np.inf, np.inf]))
            self.a=p[0]
            self.b=p[1]
            self.h=p[2]
            self.cov=cov
            self.gamma_i=0.2
            self.gamma_h=0.2
        else : 
            self.gamma_constants=False
            curve2=lambda x, a, b, h, gamma_i, gamma_h :   sirh_for_optim_normalized(x, a, b, h, self.data[2], data[0], data[1],  taking_I_into_account=self.taking_I_into_account, gamma_i=gamma_i, gamma_h=gamma_h)
            p,cov= curve_fit(curve2,np.array([i for i in range(coef*len(self.train_dates))]),obj, p0=[ 1, 1 , 5.523e-04, 0.2, 0.2],  bounds=([-np.inf, -np.inf, 0, 0, 0], [np.inf,np.inf, np.inf, np.inf, np.inf]))
            self.a=p[0]
            self.b=p[1]
            self.h=p[2]
            self.gamma_i=p[3]
            self.gamma_h=p[4]
            self.cov=cov

        
        self.trained= True
    def predict(self, reach,  alpha, method='covariance'):
        mob_predicted=np.array([self.data[2][-1] for i in range(reach)])
        reach=len(mob_predicted)
        s_0=1000000 -1
        i_0=1
        r_0=0
        h_0=0
        self.N=s_0 + i_0 + r_0 + h_0 
        S,I,R,H=run_sirh_m([s_0, i_0, r_0, h_0], self.a, self.b,self.gamma_i, self.gamma_h,   self.h ,self.data[2], 0.001)
        self.S=S
        self.I=I
        self.R=R
        self.H=H
        assert self.trained, 'The model has not been trained yet'
        # hospitalized_and_n_infected=sirh_for_optim_m(None, self.a, self.b,self.h, np.concatenate((np.array(self.data[2]), mob_predicted)))
        # hospitalized=hospitalized_and_n_infected[:len(np.array(self.data[2]))+len(mob_predicted)]
        S,I,R,H=run_sirh_m([self.S[-1], self.data[1][-1],self.N-self.data[0][-1]-self.S[-1]-self.data[1][-1],  self.data[0][-1]], self.a, self.b,self.gamma_i, self.gamma_h,   self.h ,mob_predicted, 0.001)
        self.prediction=H
        prediction=self.prediction
       
        delta_method=True
        if delta_method: 
            ci_low=[]
            ci_high=[]
            if self.gamma_constants: 
                grad=grad_theta_h_theta_m([self.S[-1], self.data[1][-1],self.N-self.data[0][-1]-self.S[-1]-self.data[1][-1],  self.data[0][-1]], [self.a, self.b , self.h], mob_predicted) # size 3 x reach
            else : 
                grad=grad_theta_h_theta_m_bis([self.S[-1], self.data[1][-1],self.N-self.data[0][-1]-self.S[-1]-self.data[1][-1],  self.data[0][-1]], [self.a, self.b ,  self.gamma_i, self.gamma_h, self.h], mob_predicted) # size 3 x reach

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
        # else: 
        #     print('sampling parameters')
        return prediction, [ci_low, ci_high]



