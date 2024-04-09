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
