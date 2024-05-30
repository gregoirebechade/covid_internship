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
from evaluate_model import evaluate_model, evaluate_model_multi, evaluate_model_multi_RMSE, evaluate_model_RMSE
import json


if __name__ =='__main__': 
    args = sys.argv # the arguments give the pandemic on which evaluate the models 
    i=int(args[1])
    j=int(args[2])
    path_to_file='all_pandemics/pandemic_'+str(i)+'_'+str(j)+'.csv'
    df=pd.read_csv(path_to_file)
    
    df.index=['n_hospitalized', 'n_infectious', 'mobility', 'R_eff']
    df.drop(columns=['Unnamed: 0'], inplace=True)
    
    n_hospitalized=np.array(df.loc['n_hospitalized'])
    n_infectious=np.array(df.loc['n_infectious'])
    mobility=np.array(df.loc['mobility'])
    r_eff=np.array(df.loc['R_eff'])
    data3D=np.array([n_hospitalized, n_infectious, mobility])




    for reach in [7, 14]: 
        print('reach', reach)

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
        dicoresults1D=dict()
        dicoresults3D=dict()


        WIS = True
        if WIS: 
            for index_points in indexs_points:
                ############### 1D
                print('index points', index_points)


                try : 
                    perf_linear=evaluate_model(model=mylinear, data=n_hospitalized, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
                except Exception as e :
                    perf_linear = np.inf
                    print('an error occured on linear')
                try : 
                    perf_bayes=evaluate_model(model=mybayes, data=n_hospitalized, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
                except Exception as e :
                    perf_bayes = np.inf
                    print('an error occured on bayes')
                


                try: 
                    perf_arima=evaluate_model(model=myarima, data=n_hospitalized, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
                except: 
                    perf_arima = np.inf
                    print('an error occured on arima')
                try: 
                    perf_exp=evaluate_model(model=myexp, data=n_hospitalized, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
                except:
                    perf_exp=np.inf
                    print('an error occured on exp')
                try: 
                    perf_moving=evaluate_model(model=mymoving, data=n_hospitalized, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights) 
                except: 
                    perf_moving = np.inf
                    print('an error occured on moving')
                try : 
                    perf_sirh1=evaluate_model(model=mysirh1, data=n_hospitalized, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
                except :
                    perf_sirh1 = np.inf
                    print('an error occured on sirh1')
                try :
                    perf_sirh2=evaluate_model(model=mysirh2, data=n_hospitalized, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
                except:
                    perf_sirh2 = np.inf
                    print('an error occured on sirh2')
                try:
                    perf_sirh3=evaluate_model(model=mysirh3, data=n_hospitalized, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
                except:
                    perf_sirh3 = np.inf
                    print('an error occured on sirh3')
                try:
                    perf_sirh4=evaluate_model(model=mysirh4, data=n_hospitalized, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
                except:
                    perf_sirh4 = np.inf
                    print('an error occured on sirh4')
                

                # # ### 3D

                try : 
                    perfmovingmulti=evaluate_model_multi(model=mymovingmulti, data=data3D, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
                except : 
                    perfmovingmulti=np.inf
                    print('an error occured on movingmulti')
                try : 
                    perfvar=evaluate_model_multi(model=myvar, data=data3D, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
                except : 
                    perfvar=np.inf
                    print('an error occured on var')
                try : 
                    perfexpmulti=evaluate_model_multi(model=myexpmulti, data=data3D, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
                except: 
                    perfexpmulti = np.inf
                    print('an error occured on exp multi')
                try : 
                    perf_sirhmulti1=evaluate_model_multi(model=mysirhmulti1, data=data3D, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
                except:
                    perf_sirhmulti1 = np.inf
                    print('an error occured on sirhmulti1')
                try :
                    perf_sirhmulti2=evaluate_model_multi(model=mysirhmulti2, data=data3D, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
                except:
                    perf_sirhmulti2 = np.inf
                    print('an error occured on sirhmulti2')
                
                print('perf_linear', perf_linear)
                print('perf_bayes', perf_bayes)
                print('perf_moving', perf_moving)
                print('perf_sirh1', perf_sirh1)
                print('perf_sirh2', perf_sirh2)
                print('perf_sirh3', perf_sirh3)
                print('perf_sirh4', perf_sirh4)
                print('perf_arima', perf_arima)
                print('perf_exp', perf_exp)
                print('perfmovingmulti', perfmovingmulti)
                print('perfvar', perfvar)
                print('perfexpmulti', perfexpmulti)
                print('perf_sirhmulti1', perf_sirhmulti1)
                print('perf_sirhmulti2', perf_sirhmulti2)


                dicoresults1D[str(index_points)]=[perf_arima,perf_exp,  perf_moving, perf_sirh1, perf_sirh2, perf_sirh3, perf_sirh4, perf_linear, perf_bayes]
                dicoresults3D[str(index_points)]=[perfvar, perfexpmulti, perfmovingmulti, perf_sirhmulti1, perf_sirhmulti2 ]


            with open('./results/global_evaluation/evaluation_with_WIS_of_3D_models_on_pandemic_'+str(i)+'_'+str(j)+'_and_reach_='+str(reach)+'.json', 'w') as f:
                json.dump(dicoresults3D, f)
            with open('./results/global_evaluation/evaluation_with_WIS_of_1D_models_on_pandemic_'+str(i)+'_'+str(j)+'_and_reach_='+str(reach)+'.json', 'w') as f:
                json.dump(dicoresults1D, f)
        

        RMSE = True
        if RMSE: 
            for index_points in indexs_points:
                ############### 1D
                print('index points', index_points)


                try : 
                    perf_linear=evaluate_model_RMSE(model=mylinear, data=n_hospitalized, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
                except Exception as e :
                    perf_linear = np.inf
                    print('an error occured on linear')
                try : 
                    perf_bayes=evaluate_model_RMSE(model=mybayes, data=n_hospitalized, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
                except Exception as e :
                    perf_bayes = np.inf
                    print('an error occured on bayes')
                


                try: 
                    perf_arima=evaluate_model_RMSE(model=myarima, data=n_hospitalized, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
                except: 
                    perf_arima = np.inf
                    print('an error occured on arima')
                try: 
                    perf_exp=evaluate_model_RMSE(model=myexp, data=n_hospitalized, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
                except:
                    perf_exp=np.inf
                    print('an error occured on exp')
                try: 
                    perf_moving=evaluate_model_RMSE(model=mymoving, data=n_hospitalized, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights) 
                except: 
                    perf_moving = np.inf
                    print('an error occured on moving')
                try : 
                    perf_sirh1=evaluate_model_RMSE(model=mysirh1, data=n_hospitalized, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
                except :
                    perf_sirh1 = np.inf
                    print('an error occured on sirh1')
                try :
                    perf_sirh2=evaluate_model_RMSE(model=mysirh2, data=n_hospitalized, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
                except:
                    perf_sirh2 = np.inf
                    print('an error occured on sirh2')
                try:
                    perf_sirh3=evaluate_model_RMSE(model=mysirh3, data=n_hospitalized, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
                except:
                    perf_sirh3 = np.inf
                    print('an error occured on sirh3')
                try:
                    perf_sirh4=evaluate_model_RMSE(model=mysirh4, data=n_hospitalized, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
                except:
                    perf_sirh4 = np.inf
                    print('an error occured on sirh4')
                

                    
                
                
                

                # # ### 3D

                try : 
                    perfmovingmulti=evaluate_model_multi_RMSE(model=mymovingmulti, data=data3D, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
                except : 
                    perfmovingmulti=np.inf
                    print('an error occured on movingmulti')
                try : 
                    perfvar=evaluate_model_multi_RMSE(model=myvar, data=data3D, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
                except : 
                    perfvar=np.inf
                    print('an error occured on var')
                try : 
                    perfexpmulti=evaluate_model_multi_RMSE(model=myexpmulti, data=data3D, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
                except: 
                    perfexpmulti = np.inf
                    print('an error occured on exp multi')
                try : 
                    perf_sirhmulti1=evaluate_model_multi_RMSE(model=mysirhmulti1, data=data3D, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
                except:
                    perf_sirhmulti1 = np.inf
                    print('an error occured on sirhmulti1')
                try :
                    perf_sirhmulti2=evaluate_model_multi_RMSE(model=mysirhmulti2, data=data3D, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
                except:
                    perf_sirhmulti2 = np.inf
                    print('an error occured on sirhmulti2')
                


                
                print('perf_linear', perf_linear)
                print('perf_bayes', perf_bayes)
                print('perf_moving', perf_moving)
                print('perf_sirh1', perf_sirh1)
                print('perf_sirh2', perf_sirh2)
                print('perf_sirh3', perf_sirh3)
                print('perf_sirh4', perf_sirh4)
                print('perf_arima', perf_arima)
                print('perf_exp', perf_exp)
                print('perfmovingmulti', perfmovingmulti)
                print('perfvar', perfvar)
                print('perfexpmulti', perfexpmulti)
                print('perf_sirhmulti1', perf_sirhmulti1)
                print('perf_sirhmulti2', perf_sirhmulti2)


                dicoresults1D[str(index_points)]=[perf_arima,perf_exp,  perf_moving, perf_sirh1, perf_sirh2, perf_sirh3, perf_sirh4, perf_linear, perf_bayes]
                dicoresults3D[str(index_points)]=[perfvar, perfexpmulti, perfmovingmulti, perf_sirhmulti1, perf_sirhmulti2 ]


            with open('./results/global_evaluation/evaluation_with_RMSE_of_3D_models_on_pandemic_'+str(i)+'_'+str(j)+'_and_reach_='+str(reach)+'.json', 'w') as f:
                    json.dump(dicoresults3D, f)
            with open('./results/global_evaluation/evaluation_with_RMSE_of_1D_models_on_pandemic_'+str(i)+'_'+str(j)+'_and_reach_='+str(reach)+'.json', 'w') as f:
                json.dump(dicoresults1D, f)
                


            