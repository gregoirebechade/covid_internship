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


class Truth(Model): 
    def train(self, train_dates, data): 
        self.index=len(data)
        self.trained=True
        self.data=data
        self.train_dates=train_dates
    def predict(self, reach, alpha): 
        prediction=new_deaths[self.index: self.index+reach]
        ic_low=prediction
        ic_high=prediction
        self.prediction=prediction
        self.ic_low=np.array(ic_low)-0.001
        self.ic_high=np.array(ic_high)+0.001
        return prediction, [ic_low, ic_high]