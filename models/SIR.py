from Model import Model





class SIRD_model(Model): 
    s_0=1000000 -1
    i_0=1
    r_0=0
    d_0=0
    dt=0.001
    def train(self, train_dates, data):
        self.data=data
        self.train_deaths=train_dates
        p,cov= curve_fit(sir_for_optim, dates_of_pandemic,data, p0=[ 5.477e-01 , 2.555e-02 , 5.523e-04],  bounds=([0,0,0], [5,5,5]))
        self.beta=p[0]
        self.gamma=p[1]
        self.d=p[2]
        self.cov=cov
        self.trained= True


    def predict(self, reach, alpha):
        assert self.trained, 'The model has not been trained yet'
        deads=sir_for_optim(self.train_deaths, self.beta, self.gamma,self.d)
        prediction =  deads[-reach:]
        perr = np.sqrt(np.diag(self.cov)) # Idea from: https://github.com/philipgerlee/Predicting-regional-COVID-19-hospital-admissions-in-Sweden-using-mobility-data.
        # we only take into account the approximation error and not the variance of the new_deaths. 
        intervals=[prediction]
        for i in range(100): 
            beta_r=np.random.normal(self.beta, perr[0], 1)[0]
            gamma_r=np.random.normal(self.gamma, perr[1], 1)[0]
            d_r=np.random.normal(self.d, perr[2], 1)[0]
            deads_sampled=sir_for_optim(np.array([i for i in range(len(self.data) + reach)]), beta_r, gamma_r,d_r)
            prediction_sampled =  deads_sampled[-reach:]
            intervals.append(prediction_sampled)
        intervals=np.array(intervals).transpose()
        ci_low=np.array([np.quantile(intervals[i], alpha/2) for i in range(reach)])
        ci_high=np.array([np.quantile(intervals[i],1-alpha/2) for i in range(reach)])
        return prediction, [ci_low, ci_high]
        

    


       
