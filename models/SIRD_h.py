from Model import Model
from useful_functions import differenciate
from SIRD import SIRD_model_2
import numpy as np


class SIRD_h(Model):
    
    def __init__(self):
        self.model=SIRD_model_2()
        self.train_dates = None
        self.trained = False
    
    def choose_model(self, gamma_constant, delta_method): 
        self.model.choose_model(gamma_constant, delta_method)

    
    def train(self, train_dates, data):
        zer=np.array([0])
        new_hospitalized=differenciate(data)
        new_data=np.concatenate((zer, new_hospitalized))
        assert len(new_data)==len(data)
        self.model.train(train_dates, new_data)
        self.trained=True
    
    def predict(self, reach, alpha): 
        prediction, intervals = self.model.predict(reach, alpha)
        prediction_summed=np.cumsum(prediction)
        int_0=np.cumsum(intervals[0])
        int_1=np.cumsum(intervals[1])

        return prediction_summed, [int_0, int_1]
