import sys
sys.path.append('./models/')
from Arima import ARIMA_Model, VAR
from SIR  import SIRD_model_2, SIRD_model, Multi_SIRD_model
from exponential_regression import ExponentialRegression, MultiDimensionalExponentialRegression
from moving_average import MovingAverage, MovingAverageMulti
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from evaluate_model import evaluate_model
import json




# EVALUATION OF MULTI DIMENSIONAL MODELS :

# importing multi_dimensional data

df_mobility=pd.read_csv('mobility.csv')
df_mobility.drop(columns=['Unnamed: 0'], inplace=True)
mobility=np.array(df_mobility['mobility'])
df = pd.read_csv('deaths_and_infections.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)
new_deaths=np.array(df['new_deaths'])
n_infected=np.array(df['n_infected'])
death_cumul=np.array([sum(new_deaths[:i]) for i in range(len(new_deaths))])
mob_shifted=np.concatenate((np.array([ 0 for i in range(17)]), mobility))
mob_17_days_ahead=(np.array([mob_shifted[i-17] for i in range(17, len(mob_shifted))]))


data =np.array([ new_deaths, n_infected, mob_17_days_ahead ])
dates_of_pandemic=np.arange(len(new_deaths))


# 7 days-ahead prediction: 



reach=7


myvar=VAR()
mysirmulti=Multi_SIRD_model()
myexpmulti=MultiDimensionalExponentialRegression()
mymovingmulti=MovingAverageMulti()

dicoresults=dict()

for index_points in indexs_points:
    print('3D, 7', index_points)
    try: 
        perf_sir=evaluate_model(model=mysirmulti, data=data, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
    except :
        perf_sir=np.inf
    try:  
        perf_exp=evaluate_model(model=myexpmulti, data=data, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
    except:
        perf_exp=np.inf
    try:
        perf_moving=evaluate_model(model=mymovingmulti, data=data, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
    except: 
        perf_moving=np.inf
    try:
        perf_arima=evaluate_model(model=myvar, data=data, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
    except:
        perf_arima=np.inf
   

    dicoresults[str(index_points)]=[perf_sir, perf_exp, perf_moving, perf_arima]
    
# write results: 
with open('comparing_models_3D_reach=7.json', 'w') as f:
    json.dump(dicoresults, f)



with open('compte_rendu_3D_reach=7.txt', 'a') as myfile: 
    print('3D, reach = 7')
    for point in dicoresults.keys(): 
        myfile.write('For the point: '+point+'\n')
        myfile.write('The best model is ' + models[(np.argmin(dicoresults[point]))]+'\n')
        myfile.write('   ')







# 14 days ahead prediction:



reach=14


myvar=VAR()
mysirmulti=Multi_SIRD_model()
myexpmulti=MultiDimensionalExponentialRegression()
mymovingmulti=MovingAverageMulti()

dicoresults=dict()

for index_points in indexs_points:
    print('3D, 14', index_points)
    try: 
        perf_sir=evaluate_model(model=mysirmulti, data=data, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
    except :
        perf_sir=np.inf
    try:  
        perf_exp=evaluate_model(model=myexpmulti, data=data, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
    except:
        perf_exp=np.inf
    try:
        perf_moving=evaluate_model(model=mymovingmulti, data=data, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
    except: 
        perf_moving=np.inf
    try:
        perf_arima=evaluate_model(model=myvar, data=data, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
    except:
        perf_arima=np.inf
   

    dicoresults[str(index_points)]=[perf_sir, perf_exp, perf_moving, perf_arima]
    
# write results: 
with open('comparing_models_3D_reach=14.json', 'w') as f:
    json.dump(dicoresults, f)



with open('compte_rendu_3D_reach=14.txt', 'a') as myfile: 
    print('3D, reach = 14')
    for point in dicoresults.keys(): 
        myfile.write('For the point: '+point+'\n')
        myfile.write('The best model is ' + models[(np.argmin(dicoresults[point]))]+'\n')
        myfile.write('   ')





## EVALUATION OF MONO DIMENSIONAL MODELS : 

# importing mono_dimensional data

df = pd.read_csv('deaths_and_infections.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)
new_deaths=np.array(df['new_deaths'])
death_cumul=np.array([sum(new_deaths[:i]) for i in range(len(new_deaths))])
dates_of_pandemic=np.arange(len(new_deaths))


# global values for the whole evaluation: 


alphas=np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
indexs_points=[[30],[35], [40], [45], [50],[55], [60],[65],  [70],[75],  [80],[85], [90],[95], [100],[105] [110]]
weights=np.concatenate(([0.5], alphas * 0.5))
models = ['SIRD', 'SIRD2', 'ExponentialRegression', 'MovingAverage', 'Arima']




reach=7


myarima=ARIMA_Model()
mysir2=SIRD_model_2()
mysir=SIRD_model()
myexp=ExponentialRegression()
mymoving=MovingAverage()
dicoresults=dict()

for index_points in indexs_points:
    print('1D, 7', index_points)
    try: 
        perf_sir=evaluate_model(model=mysir, data=new_deaths, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
    except :
        perf_sir=np.inf
    try:  
        perf_exp=evaluate_model(model=myexp, data=new_deaths, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
    except:
        perf_exp=np.inf
    try:
        perf_moving=evaluate_model(model=mymoving, data=new_deaths, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
    except: 
        perf_moving=np.inf
    try:
        perf_arima=evaluate_model(model=myarima, data=new_deaths, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
    except:
        perf_arima=np.inf
    try: 
        perf_sir2=evaluate_model(model=mysir2, data=new_deaths, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
    except:
        perf_sir2=np.inf

    dicoresults[str(index_points)]=[perf_sir, perf_exp, perf_moving, perf_arima]
    
# write results: 
with open('comparing_models_1D_reach=7.json', 'w') as f:
    json.dump(dicoresults, f)



with open('compte_rendu_1D_reach=7.txt', 'a') as myfile: 
    print('1D, reach = 7')
    for point in dicoresults.keys(): 
        myfile.write('For the point: '+point+'\n')
        myfile.write('The best model is ' + models[(np.argmin(dicoresults[point]))]+'\n')
        myfile.write('   ')





myarima=ARIMA_Model()
mysir2=SIRD_model_2()
myexp=ExponentialRegression()
mymoving=MovingAverage()
mysir=SIRD_model()
reach=14
dicoresults=dict()


for index_points in indexs_points:
    try: 
        perf_sir=evaluate_model(model=mysir, data=new_deaths, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
    except :
        perf_sir=np.inf
    try:  
        perf_exp=evaluate_model(model=myexp, data=new_deaths, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
    except:
        perf_exp=np.inf
    try:
        perf_moving=evaluate_model(model=mymoving, data=new_deaths, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
    except: 
        perf_moving=np.inf
    try:
        perf_arima=evaluate_model(model=myarima, data=new_deaths, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
    except:
        perf_arima=np.inf
    try:
        perf_sir2=evaluate_model(model=mysir2, data=new_deaths, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
    except:
        perf_sir2=np.inf

    dicoresults[str(index_points)]=[perf_sir, perf_exp, perf_moving, perf_arima]
    
# write results: 
with open('comparing_models_1D_reach=14.json', 'w') as f:
    json.dump(dicoresults, f)


with open('compte_rendu_1D_reach=14.txt', 'a') as myfile: 
    for point in dicoresults.keys(): 
        myfile.write('For the point: '+point+'\n')
        myfile.write('The best model is ' + models[(np.argmin(dicoresults[point]))]+'\n')
        myfile.write('   ')


