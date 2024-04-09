import numpy as np
from Model import Model



def IS(interval : tuple, point : float, alpha: float) -> float: 
    assert interval[0] < interval[1]
    assert alpha >= 0
    assert alpha <= 1
    l=interval[0]
    u = interval[1]
    dispersion = u-l
    
    if point < l : 
        underprediction= (2/alpha)*(l-point)
    else: 
        underprediction=0
    if point > u :
        overprediction = (2/alpha)*(point-u)
    else: 
        overprediction=0
    
    return underprediction + overprediction + dispersion



def WIS(prediction: float, interval : tuple, point_of_evaluation : float, alphas: list, weights: list) -> float:
    """

    WIS computes the Weighted Interval Score of a model that predicts a point and a list of confidence intervals.
    The fuction taks as an input a prediction, a list of confidence intervals of precision alpha, a list of weights to apply to the different intervals and a point to evaluate the prediction on.
    
    """ 
    assert interval[0] < interval[1]
    assert all([alpha >= 0 for alpha in alphas])
    assert all([alpha <= 1 for alpha in alphas])
    K = len(alphas)
    loss=0
    for k in range(1, K): 
        
        loss += weights[k]*IS(interval, point_of_evaluation, alphas[k])
    loss += weights[0]* abs(prediction - point_of_evaluation)
    return loss


def evaluate_model(model: Model, data: np.array, alphas: list, evaluation_point_indexs: list, reach: int, weights: list) -> float: 
    
    loss=0

    for index in evaluation_point_indexs: 
        model.train(train_dates = [i for i in range(index)], data = data[:index] )
        for alpha in alphas: 
            prediction, intervals = model.predict(reach, alpha)
            prediction=prediction[-1]
            interval_low=intervals[0][-1]
            interval_high=intervals[1][-1]
            wis=WIS(prediction=prediction, interval = (interval_low, interval_high), point_of_evaluation = data[index+reach], alphas = alphas , weights = weights)
            loss+=wis
    return loss / len(evaluation_point_indexs) # average loss over all evaluation points