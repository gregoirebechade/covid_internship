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
    

    def train(self, dates_of_pandemic_train, datatrain): 
        n_days=min(len(dates_of_pandemic_train), 30)
        br=BayesianRidge()
        data=[]
        y=[]
        for i in range(n_days, len(datatrain)): 
            data.append(datatrain[i-n_days:i])
            y.append(datatrain[i])
        data=np.array(data)
        self.mydata=data
        self.results=br.fit(data, y)
        self.model=br
        self.data=datatrain
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

    def train(self, dates_of_pandemic_train, datatrain): 
        self.model=BayesianRegressionModel()
        self.model.train(dates_of_pandemic_train, datatrain[0])
        self.trained=True
        self.data=datatrain
    def predict(self, reach, alpha): 
        prediction, intervals=self.model.predict(reach, alpha)
        return prediction, intervals