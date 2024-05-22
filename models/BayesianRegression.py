import numpy as np 
from Model import Model, Multi_Dimensional_Model
import scipy.stats
from useful_functions import predict_model
from sklearn.linear_model import BayesianRidge

def grad_theta_h_theta(h, theta, x): 
    grad=np.zeros(len(theta))
    for i in range(len(theta)): 
        theta_plus=theta.copy()
        theta_plus[i]=theta[i]+0.0001
        grad[i]=(h(theta_plus, x)-h(theta, x))/0.0001
    return grad


def create_br_model(coefs): 

    br=BayesianRidge()
    br.fit(np.array([0 for i in range(len(coefs))]).reshape(1, -1), [1])
    br.coef_=coefs
    return br

def prediction_br_model(theta, x ): 
    br = create_br_model(theta)
    return br.predict(x.reshape(1, -1))[0]

class BayesianRegressionModel(Model):
    

    def train(self, train_dates, data): 
        n_days=30
        if n_days > len(train_dates)/2: 
            n_days = len(train_dates)//2 -1 # we avoid to have a RL with more dimensions thanh the number of points to avoid negative sigma2
        
        br=BayesianRidge()
        data_ml=[]
        y=[]
        for i in range(n_days, len(data)): 
            data_ml.append(data[i-n_days:i])
            y.append(data[i])
        data_ml=np.array(data_ml)
        self.mydata=data_ml
        print('data_ml', data_ml)
        print('y', y)
        self.results=br.fit(data_ml, y)
        self.model=br
        self.data=data
        self.trained=True
        self.y=y
    

    def predict(self, reach, alpha): 
        prediction = predict_model(self.model, self.mydata, self.y, reach)
        covariance_matrix=self.model.sigma_
        last_day=self.mydata[-1]
        last_y=self.y[-1]
        ci_inf=[]
        ci_up=[]
        x=np.concatenate((last_day[1:], np.array([last_y]))).reshape(len(last_day), 1)
        self.x=x
        self.covariance_matrix=covariance_matrix    
        self.grads= []
        self.varps=[]
        for i in range(reach):
            grad=grad_theta_h_theta(prediction_br_model, self.model.coef_, x)
            varp=np.matmul(grad.T, np.matmul(covariance_matrix, grad))
            ci_inf.append(scipy.stats.norm.ppf(alpha/2, loc=prediction[i], scale=np.sqrt(varp))[0])
            ci_up.append(scipy.stats.norm.ppf(1-alpha/2, loc=prediction[i], scale=np.sqrt(varp))[0])
            self.grads.append(grad)
            self.varps.append(varp)
            x=np.concatenate((x[1:], prediction[i].reshape(1, 1)))
        self.grads=np.array(self.grads)
        return prediction, np.array([ci_inf, ci_up]).reshape(2, reach)



        
class MultiDimensionalBayesianRegression(Multi_Dimensional_Model): 

    def train(self, train_dates, data): 
        self.model=BayesianRegressionModel()
        self.model.train(train_dates, data[0])
        self.trained=True
        self.data=data
    def predict(self, reach, alpha): 
        prediction, intervals=self.model.predict(reach, alpha)
        return prediction, intervals