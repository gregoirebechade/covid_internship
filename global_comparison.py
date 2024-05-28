import sys
sys.path.append('./models/')
from Arima import ARIMA_Model, VAR_m
from exponential_regression import ExponentialRegression, MultiDimensionalExponentialRegression
from SIRH  import *
from LinearRegression import *
from BayesianRegression import *
from moving_average import MovingAverage, MovingAverageMulti
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from evaluate_model import evaluate_model, evaluate_model_multi, evaluate_model_multi_RMSE, evaluate_model_RMSE
import json


all_pandemics=pd.read_csv('all_pandemics.csv')
all_pandemics.drop(columns=['Unnamed: 0'], inplace=True)
all_pandemics=np.array(all_pandemics)

if __name__ =='__main__': 
    args = sys.argv
    min=int(args[1])
    max=int(args[2])