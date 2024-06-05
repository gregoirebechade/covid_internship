import sys
sys.path.append('./models/')
from Arima import ARIMA_Model, VAR_m
from exponential_regression import ExponentialRegression, MultiDimensionalExponentialRegression
from SIRH  import SIRH_model_2, Multi_SIRH_model
from LinearRegression import LinearRegressionModel
from BayesianRegression import  BayesianRegressionModel
from moving_average import MovingAverage, MovingAverageMulti
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from evaluate_model import WIS
import json


if __name__ =='__main__': 
    args = sys.argv # the arguments give the pandemic on which evaluate the models 
    mob_of_the_pandemic=int(args[1])
    number_of_the_pandemic=int(args[2])
    path_to_file='all_pandemics/pandemic_'+str(mob_of_the_pandemic)+'_'+str(number_of_the_pandemic)+'.csv'
    df=pd.read_csv(path_to_file)
    
    df.index=['n_hospitalized', 'n_infectious', 'mobility', 'R_eff']
    df.drop(columns=['Unnamed: 0'], inplace=True)
    
    n_hospitalized=np.array(df.loc['n_hospitalized'])
    n_infectious=np.array(df.loc['n_infectious'])
    mobility=np.array(df.loc['mobility'])
    r_eff=np.array(df.loc['R_eff'])
    data3D=np.array([n_hospitalized, n_infectious, mobility])



    myarima=ARIMA_Model()
    myexp=ExponentialRegression()
    myexpmulti=MultiDimensionalExponentialRegression()
    mymoving=MovingAverage()
    mysirh1=SIRH_model_2()
    mysirh1.choose_model(True, True, True)
    mysirh2=SIRH_model_2()
    mysirh2.choose_model(True, False, True)
    mysirh3=SIRH_model_2()
    mysirh3.choose_model(False, True, True)
    mysirh4=SIRH_model_2()
    mysirh4.choose_model(False, False, True)
    mylinear=LinearRegressionModel()
    mybayes=BayesianRegressionModel()
    mysirhmulti1=Multi_SIRH_model()
    mysirhmulti1.choose_model(True, True)
    mysirhmulti2=Multi_SIRH_model()
    mysirhmulti2.choose_model(True, False)
    myvar=VAR_m()
    mymovingmulti=MovingAverageMulti()
    alphas=np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    indexs_points=[[20*i] for i in range(1, 15) ] 
    weights=np.concatenate((np.array([0.5]), alphas * 0.5))

    models1Dnames=['ARIMA', 'Exponential', 'Moving Average', 'SIRH1', 'SIRH2', 'SIRH3', 'SIRH4', 'Linear Regression', 'Bayesian Regression']
    models3Dnames=[ 'VAR', 'Exponential Multi', 'Moving Average Multi', 'SIRH Multi1', 'SIRH Multi2']

    models1D=[myarima, myexp, mymoving, mysirh1, mysirh2, mysirh3, mysirh4, mylinear, mybayes]
    models3D=[myvar, myexpmulti, mymovingmulti, mysirhmulti1, mysirhmulti2]


    dico_wis_1D_reach_7=dict()
    for point in indexs_points: 
        dico_wis_1D_reach_7[str(point)]=[]
    dico_wis_1D_reach_14=dict()
    for point in indexs_points:
        dico_wis_1D_reach_14[str(point)]=[]
    dico_rmse_1D_reach_7=dict()
    for point in indexs_points:
        dico_rmse_1D_reach_7[str(point)]=[]
    dico_rmse_1D_reach_14=dict()
    for point in indexs_points:
        dico_rmse_1D_reach_14[str(point)]=[]
    dico_wis_3D_reach_7=dict()
    for point in indexs_points:
        dico_wis_3D_reach_7[str(point)]=[]
    dico_wis_3D_reach_14=dict()
    for point in indexs_points:
        dico_wis_3D_reach_14[str(point)]=[]
    dico_rmse_3D_reach_7=dict()
    for point in indexs_points:
        dico_rmse_3D_reach_7[str(point)]=[]
    dico_rmse_3D_reach_14=dict()
    for point in indexs_points:
        dico_rmse_3D_reach_14[str(point)]=[]




    df_results_7=pd.DataFrame(columns=models1Dnames + models3Dnames + ['Real values'])
    df_results_14=pd.DataFrame(columns=models1Dnames + models3Dnames + ['Real values'])

    for point in indexs_points: 

        
        prediction_7=[]
        prediction_14=[]


        for model in models1D:
            try : 
                model.train(train_dates = [i for i in range(point[0])], data = n_hospitalized[:point[0]])
            except:
                if reach ==7 : 
                    dico_wis_1D_reach_7[str(point)].append(np.inf)
                    dico_rmse_1D_reach_7[str(point)].append(np.inf)
                else:
                    dico_wis_1D_reach_14[str(point)].append(np.inf)
                    dico_rmse_1D_reach_14[str(point)].append(np.inf)
            for reach in [7, 14]:
                intervals=[]
                for alpha in alphas:
                    try : 
                        prediction, interval = model.predict(reach, alpha)
                        interval_low=interval[0][-1]
                        interval_high=interval[1][-1]
                         
                        prediction=prediction[-1]
                    except : 
                        interval_low=0
                        interval_high=0
                        prediction=np.inf
                    intervals.append((interval_low, interval_high))

                if reach ==7 : 
                    prediction_7.append(prediction)

                else:
                    prediction_14.append(prediction)
                

                if prediction ==np.inf: 
                    wis=np.inf
                    RMSE=np.inf
                else :
                    try : 
                        wis=WIS(prediction=prediction, intervals = intervals, point_of_evaluation = n_hospitalized[point[0]+reach-1], alphas = alphas , weights = weights)
                    except : 
                        wis=np.inf
                    try : 
                        RMSE=np.sqrt((prediction - n_hospitalized[point[0]+reach-1])**2)
                    except : 
                        RMSE=np.inf
                if reach ==7 : 
                    dico_wis_1D_reach_7[str(point)].append(wis)
                    dico_rmse_1D_reach_7[str(point)].append(RMSE)
                else:
                    dico_wis_1D_reach_14[str(point)].append(wis)
                    dico_rmse_1D_reach_14[str(point)].append(RMSE)
        

        for model in models3D:
                try: 
                    model.train(train_dates = [i for i in range(point[0])], data = data3D[:,:point[0]])
                except:
                    if reach ==7 : 
                        dico_wis_3D_reach_7[str(point)].append(np.inf)
                        dico_rmse_3D_reach_7[str(point)].append(np.inf)
                    else:
                        dico_wis_3D_reach_14[str(point)].append(np.inf)
                        dico_rmse_3D_reach_14[str(point)].append(np.inf)
                for reach in [7, 14]:
                    intervals=[]
                    for alpha in alphas:
                        try: 
                            prediction, interval = model.predict(reach, alpha)
                            interval_low=interval[0][-1]
                            interval_high=interval[1][-1]
                            prediction=prediction[-1]

                        except : 
                            interval_low=0
                            interval_high=0
                            prediction=np.inf
                        intervals.append((interval_low, interval_high)) 

                    if reach ==7 :
                        prediction_7.append(prediction)
                    
                    else:
                        prediction_14.append(prediction)
                    
                           
                    if prediction == np.inf:
                        wis=np.inf
                        RMSE=np.inf
                    else : 
                        try : 
                            wis=WIS(prediction=prediction, intervals = intervals, point_of_evaluation = n_hospitalized[point[0]+reach-1], alphas = alphas , weights = weights)
                        except : 
                            wis=np.inf
                        try :
                             
                            RMSE=np.sqrt((prediction - n_hospitalized[point[0]+reach-1])**2)
                        except :
                            RMSE=np.inf
    
                    if reach ==7 : 
                        dico_wis_3D_reach_7[str(point)].append(wis)
                        dico_rmse_3D_reach_7[str(point)].append(RMSE)
                    else:
                        dico_wis_3D_reach_14[str(point)].append(wis)
                        dico_rmse_3D_reach_14[str(point)].append(RMSE)
        df_results_7.loc[str(point[0])] = prediction_7 + [n_hospitalized[point[0]+7-1]]
        df_results_14.loc[str(point[0])] = prediction_14 + [n_hospitalized[point[0]+14-1]]
    

    # write results : 

    reach=7
    with open('./results/global_evaluation_bis/evaluation_with_RMSE_of_3D_models_on_pandemic_'+str(mob_of_the_pandemic)+'_'+str(number_of_the_pandemic)+'_and_reach_='+str(reach)+'.json', 'w') as f:
            json.dump(dico_rmse_3D_reach_7, f)
    
    with open('./results/global_evaluation_bis/evaluation_with_WIS_of_3D_models_on_pandemic_'+str(mob_of_the_pandemic)+'_'+str(number_of_the_pandemic)+'_and_reach_='+str(reach)+'.json', 'w') as f:
            json.dump(dico_wis_3D_reach_7, f)

    with open('./results/global_evaluation_bis/evaluation_with_RMSE_of_1D_models_on_pandemic_'+str(mob_of_the_pandemic)+'_'+str(number_of_the_pandemic)+'_and_reach_='+str(reach)+'.json', 'w') as f:
            json.dump(dico_rmse_1D_reach_7, f)

    with open('./results/global_evaluation_bis/evaluation_with_WIS_of_1D_models_on_pandemic_'+str(mob_of_the_pandemic)+'_'+str(number_of_the_pandemic)+'_and_reach_='+str(reach)+'.json', 'w') as f:
            json.dump(dico_wis_1D_reach_7, f)
    

    reach=14

    with open('./results/global_evaluation_bis/evaluation_with_RMSE_of_3D_models_on_pandemic_'+str(mob_of_the_pandemic)+'_'+str(number_of_the_pandemic)+'_and_reach_='+str(reach)+'.json', 'w') as f:
            json.dump(dico_rmse_3D_reach_14, f)
    
    with open('./results/global_evaluation_bis/evaluation_with_WIS_of_3D_models_on_pandemic_'+str(mob_of_the_pandemic)+'_'+str(number_of_the_pandemic)+'_and_reach_='+str(reach)+'.json', 'w') as f:
            json.dump(dico_wis_3D_reach_14, f)

    with open('./results/global_evaluation_bis/evaluation_with_RMSE_of_1D_models_on_pandemic_'+str(mob_of_the_pandemic)+'_'+str(number_of_the_pandemic)+'_and_reach_='+str(reach)+'.json', 'w') as f:
            json.dump(dico_rmse_1D_reach_14, f)

    with open('./results/global_evaluation_bis/evaluation_with_WIS_of_1D_models_on_pandemic_'+str(mob_of_the_pandemic)+'_'+str(number_of_the_pandemic)+'_and_reach_='+str(reach)+'.json', 'w') as f:
            json.dump(dico_wis_1D_reach_14, f)

    df_results_7.to_csv('./results/predictions_of_the_models/predictions_7_days_on_pandemic_'+str(mob_of_the_pandemic)+'_'+str(number_of_the_pandemic)+'.csv')
    df_results_14.to_csv('./results/predictions_of_the_models/predictions_14_days_on_pandemic_'+str(mob_of_the_pandemic)+'_'+str(number_of_the_pandemic)+'.csv')