import matplotlib.pyplot as plt
import numpy as np

class Model: 
    def __init__(self) :
        self.trained=False
    def reinitialize(self): 
        self.trained=False
        self.data=None
        self.train_dates=None
        self.model=None
    def train(self, train_dates, data):
        self.train_dates=train_dates
        self.data=data # to be implemented in the child class
    def predict(self, reach, alphas):
        pass # to be implemented in the child class 

    def plot(self, reach, alpha, title=None): 
        assert self.trained, 'The model has not been trained yet'
        prediction, intervals = self.predict(reach, alpha)
        ci_low=[max(0, intervals[0][i]) for i in range(len(intervals[0]))]
        ci_high=intervals[1]
        plt.plot([i for i in range(len(self.data))], self.data, label='real data')
        plt.plot([i for i in range(len(self.data), len(self.data) + reach)] , prediction, label='forecast ')
        plt.fill_between([i for i in range(len(self.data), len(self.data) + reach)], ci_low, ci_high, color='black', alpha=.3, label='confidence interval at ' + str(round((1-alpha)*100)) + '%')
        plt.legend()
        plt.axvline(len(self.data), linestyle='--')
        plt.xlim(0,len(self.data)+reach)
        plt.title(title)
        plt.show()
