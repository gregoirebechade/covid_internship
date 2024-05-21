import numpy as np 
from Model import Model
from sklearn.linear_model import LinearRegression
import scipy.stats



def predict_model(model, data_train,y_train,  reach): 
    prediction_reach_days_ahead=[]
    a=np.concatenate((data_train[-1], np.array([y_train[-1]])))[1:]
    predict=model.predict(a.reshape(1, -1))
    prediction_reach_days_ahead.append(predict)
    for i in range(1, reach):
        a=np.concatenate((a[1:], predict))
        predict=model.predict(a.reshape(1, -1))
        prediction_reach_days_ahead.append(predict)
    return np.array(prediction_reach_days_ahead)

def estimate_sigma2(linear_regression, data, y): 
    predictions = linear_regression.predict(data)
    d=linear_regression.coef_.shape[0]
    n=len(y)
    sigma2=np.sum((predictions -y )**2)/(n-d) # unbiased estimator of the variance of the noise
    return sigma2


def computa_covariance_matrix(linear_regression, data, y): 
    sigma2=estimate_sigma2(linear_regression, data, y)
    # X=np.concatenate((np.ones((len(data), 1)), data), axis=1)
    X=data
    return np.linalg.inv(np.dot(X.T, X))*sigma2

class LinearRegressionModel(Model):
    

    def train(self, dates_of_pandemic_train, datatrain): 
        n_days=min(len(dates_of_pandemic_train), 30)
        lr=LinearRegression()
        data=[]
        y=[]
        for i in range(n_days, len(datatrain)): 
            data.append(datatrain[i-n_days:i])
            y.append(datatrain[i])
        data=np.array(data)
        self.mydata=data
        self.results=lr.fit(data, y)
        self.model=lr
        self.data=datatrain
        self.trained=True
        self.y=y
    

    def predict(self, reach, alpha): 
        prediction = predict_model(self.model, self.mydata, self.y, reach)
        covariance_matrix=computa_covariance_matrix(self.model, self.mydata, self.y)
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
        return prediction, np.array([ci_inf, ci_up]).reshape(2, 7)



        
           