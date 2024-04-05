import numpy as np
import pandas as pd
import json
df = pd.read_csv('deaths_and_infections.csv')
# remove a columns from a df: 
df.drop(columns=['Unnamed: 0'], inplace=True)


new_deaths=np.array(df['new_deaths'])
death_cumul=np.array([sum(new_deaths[:i]) for i in range(len(new_deaths))])
dates_of_pandemic=np.arange(len(new_deaths))


s_0=1000000 -1
i_0=1
r_0=0
d_0=0
beta=3
gamma=0.5
d=0.1
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
    return differenciate(np.concatenate((zer,d_arr)))




betas=np.arange(0, 31, 0.5)
gammas=np.arange(0,31, 0.5)
ds=np.arange(0, 31, 0.5)
min=np.sum(np.abs(sir_for_optim(dates_of_pandemic,  0.5, 0.6, 0.3) - new_deaths))
print(min)
dicoresults=dict()
gammamin=0
betamin=0
dmin=0


print('beginning grid search')

for beta in betas: 
    for gamma in gammas: 
        for d in ds:
            error = np.sum(np.abs(sir_for_optim(dates_of_pandemic, beta, gamma, d) - new_deaths))
            if error<min: 
                min=error
                gammamin=gamma
                betamin=beta
                dmin=d
                print('new min found')
                print(min)
                print('the values are : ')
                print('beta = ', beta)
                print('gamma = ', gamma)
                print('d = ', d)
            dicoresults['beta = '+ str(beta) + ' gamma = '+ str(gamma) + ' d = '+ str(d)]=min
            

with open('results.json', 'w') as f:
    json.dump(dicoresults, f)
print('saved')