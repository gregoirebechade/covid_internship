import numpy as np 
from Model import Model, Multi_Dimensional_Model
from sklearn.linear_model import LinearRegression
import scipy.stats
from useful_functions import predict_model



def estimate_sigma2(linear_regression, data, y): 
    predictions = linear_regression.predict(data)
    d=linear_regression.coef_.shape[0]
    n=len(y)
    sigma2=np.sum((predictions -y )**2)/(n-d) # unbiased estimator of the variance of the noise
    return sigma2


def compute_covariance_matrix(linear_regression, data, y): 
    sigma2=estimate_sigma2(linear_regression, data, y)
    X=data
    return np.linalg.inv(np.dot(X.T, X))*sigma2

class LinearRegressionModel(Model):
    

    def train(self, train_dates, data): 
        n_days=30
        if n_days >= len(train_dates)/2: 
            n_days = len(train_dates)//2 -1 # we avoid to have a RL with more dimensions thanh the number of points to avoid negative
        
        lr=LinearRegression()
        data_for_ml=[]
        y=[]
        for i in range(n_days, len(data)): 
            data_for_ml.append(data[i-n_days:i])
            y.append(data[i])
        data_for_ml=np.array(data_for_ml)
        self.mydata=data_for_ml
        self.results=lr.fit(data_for_ml, y)
        self.model=lr
        self.data=data
        self.trained=True
        self.y=y



    def predict(self, reach, alpha): 
        prediction = predict_model(self.model, self.mydata, self.y, reach)
        covariance_matrix=compute_covariance_matrix(self.model, self.mydata, self.y)
        last_day=self.mydata[-1]
        last_y=self.y[-1]
        ci_inf=[]
        ci_up=[]
        x=np.concatenate((last_day[1:], np.array([last_y]))).reshape(len(last_day), 1)
        self.x=x
        self.covariance_matrix=covariance_matrix    
        for i in range(reach):
            varp=np.matmul(x.T ,np.matmul( covariance_matrix , x))
            ci_inf.append(scipy.stats.norm.ppf(alpha/2, loc=prediction[i], scale=np.sqrt(varp))[0])
            ci_up.append(scipy.stats.norm.ppf(1-alpha/2, loc=prediction[i], scale=np.sqrt(varp))[0])
            x=np.concatenate((x[1:], prediction[i].reshape(1, 1)))
        return prediction.reshape(reach, ), list(np.array([ci_inf, ci_up]).reshape(2, reach))



        
class MultiDimensionalLinearRegression(Multi_Dimensional_Model): 

    def train(self, train_dates, data): 
        self.model=LinearRegressionModel()
        self.model.train(train_dates, data[0])
        self.trained=True
        self.data=data
    def predict(self, reach, alpha): 
        prediction, intervals=self.model.predict(reach, alpha)
        return prediction.reshape(reach,), intervals