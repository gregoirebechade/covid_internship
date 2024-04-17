from Model import Model
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import numpy as np

class ARIMA_Model(Model):

    def train(self, train_dates, data, grid_search=False, p=3, d=0, q=3): 
        self.data=data
        if grid_search:
            min = 1000
            pmin=0
            dmin=0
            qmin=0
            for p in range(7): 
                for d in range(4): 
                    for q in range(7): 
                        if not (p,d,q) == (0,1,0):
                            model = ARIMA(data, order=(p,d,q))  
                            try : 
                                fitted = model.fit()  
                                prediction=fitted.predict(start=1, end=len(data) , typ='levels')
                                error = np.mean(np.abs(prediction - data))
                                if error < min:
                                    min = error
                                    pmin=p
                                    dmin=d
                                    qmin=q
                            except np.linalg.LinAlgError as err : 
                                pass
        else :
            pmin=p
            dmin=d
            qmin=q
        self.model = ARIMA(data, order=(pmin,dmin,qmin))
        self.p=pmin
        self.d=dmin
        self.q=qmin
        self.fitted=self.model.fit()
        self.trained= True
    
    def predict(self, reach, alpha):
        assert self.trained, 'The model has not been trained yet'
        predifore=self.fitted.get_forecast(reach).predicted_mean # this is an array of size reach
        assert type(predifore) == np.ndarray, 'The prediction should be a numpy array'
        # confidence_intervals=[self.fitted.get_forecast(reach).conf_int(alpha=alp) for alp in alphas]
        interval = self.fitted.get_forecast(reach).conf_int(alpha=alpha)
        ci_low=[max(elt[0],0) for elt in interval]
        ci_high=[elt[1] for elt in interval]
        return predifore, [ci_low, ci_high]
    
    # def plot(self, reach, alpha): 
    #     assert self.trained, 'The model has not been trained yet'
    #     prediction, intervals = self.predict(reach, alpha)
    #     ci_low=intervals[0]
    #     ci_high=intervals[1]
    #     plt.plot([i for i in range(len(self.data))], self.data, label='real data')
    #     plt.plot([i for i in range(len(self.data), len(self.data) + reach)] , prediction, label='forecast ')
    #     plt.fill_between([i for i in range(len(self.data), len(self.data) + reach)], ci_low, ci_high, color='black', alpha=.3, label='confidence interval at ' + str(round((1-alpha)*100)) + '%')
    #     plt.legend()
    #     plt.axvline(len(self.data), linestyle='--')
    #     plt.xlim(0,len(self.data)+reach)
    #     plt.show()
        