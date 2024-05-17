from Model import Model
from useful_functions import differenciate
from SIRD import SIRD_model_2


class SIRD_h(Model):
    
    def __init__(self):
        self.model=SIRD_model_2()
        self.train_dates = None
        self.trained = False
    
    def choose_model(self, gamma_constant, delta_method): 
        self.model = 'monotonic'
    
    def train(self, train_dates, data):
        self.train_dates = train_dates
        self.trained = True
