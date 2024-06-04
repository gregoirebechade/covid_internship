import numpy as np
from scipy.stats import wasserstein_distance



def differenciate(x): 
    dx=[x[i+1]-x[i] for i in range(len(x)-1)]
    return dx


def shift(x: np.array, n:float): 
    if n >0 : 
        return np.concatenate((np.array([ x[0] for i in range(int(n))]), x))[:len(x)] # we assume that the n first values are the same as the first value of the array
    elif n < 0 :
        return np.concatenate((x, np.array([ x[-1] for i in range(int(-n))])))[-len(x):]
    else :
        return x
    

def plot_predictions(models: list, data: np.array, dates_of_pandemic: np.array, reach: int, points_of_evaluation: list, fig, ax):
    new_deaths, n_infected, mobility = data
    ax.plot(dates_of_pandemic, new_deaths, c='black')
    colours=['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for j in range(len(points_of_evaluation)) : 
        point = points_of_evaluation[j]
        for i in range(len(models)): 
            model=models[i]
            if model.type=='1D': 
                model.train( dates_of_pandemic[:point], new_deaths[:point])
                print(model.name)
                try: 
                    pred, ints = model.predict(reach, 0.05)
                    if j ==0: 
                        ax.plot(np.arange(point, point+reach), pred, c=colours[i], label=model.name)
                    else: 
                        ax.plot(np.arange(point, point+reach), pred, c=colours[i] )
                except: 
                    print('oups')
            if model.type=='3D': 
                model.train(dates_of_pandemic[:point],np.array([new_deaths[:point], n_infected[:point], mobility[:point]]))
                pred, ints=model.predict(7, 0.05)
                if j ==0: 
                    ax.plot(np.arange(point, point+reach), pred, c=colours[i], label=model.name)
                else: 
                    ax.plot(np.arange(point, point+reach), pred, c=colours[i])
    ax.legend()




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





def diff_between_2_arrays(array1, array2): # this function punishes the pandemics when they do not have the same amplitude (difference in maximum), but also when their derivatives do not have the same amplitude, and when their second derivatives do not have the same amplitude
    derive1=np.array(differenciate(array1))
    derive2=np.array(differenciate(array2))
    derivee1=np.array(differenciate(derive1))
    derivee2=np.array(differenciate(derive2))
    max1=max(array1)
    max2=max(array2)
    maxder1=max(derive1)
    maxder2=max(derive2)
    maxderder1=max(derivee1)
    maxderder2=max(derivee2)
    ar1_normalized=array1/np.sum(abs(array1))
    ar2_normalized=array2/np.sum(abs(array2))
    der1_normalized=derive1/np.sum(abs(derive1))
    der2_normalized=derive2/np.sum(abs(derive2))
    derder1_normalized=derivee1/np.sum(abs(derivee1))
    derder2_normalized=derivee2/np.sum(abs(derivee2))
    res=[]
    if max1>max2: 
        res.append(max1/max2 -1) # difference of amplitude of the two arrays
    else : 
        res.append(max2/max1-1)
    if maxder1>maxder2:
        res.append(maxder1/maxder2-1) # difference of amplitude of the two derivatives
    else :
        res.append(maxder2/maxder1-1)
    if maxderder1>maxderder2:
        res.append(maxderder1/maxderder2-1) # difference of amplitude of the two second derivatives
    else :
        res.append(maxderder2/maxderder1-1)
    res.append(np.sum([abs(ar1_normalized[i]-ar2_normalized[i]) for i in range(len(ar1_normalized))])) # absolute difference of the two arrays
    res.append(np.sum([abs(der1_normalized[i]-der2_normalized[i]) for i in range(len(der1_normalized))])) # absolute difference of the two derivatives
    res.append(np.sum([abs(derder1_normalized[i]-derder2_normalized[i]) for i in range(len(derder1_normalized))])) # absolute difference of the two second derivatives
    res=np.array(res)
    return np.sum(res**2)
            

def diff_between_2_arrays_2(array1, array2): # same function but with wassertsein distance instead of absolute difference
    derive1=np.array(differenciate(array1))
    derive2=np.array(differenciate(array2))
    derivee1=np.array(differenciate(derive1))
    derivee2=np.array(differenciate(derive2))
    max1=max(array1)
    max2=max(array2)
    maxder1=max(derive1)
    maxder2=max(derive2)
    maxderder1=max(derivee1)
    maxderder2=max(derivee2)
    ar1_normalized=array1/np.sum(abs(array1))
    ar2_normalized=array2/np.sum(abs(array2))
    der1_normalized=derive1/np.sum(abs(derive1))
    der2_normalized=derive2/np.sum(abs(derive2))
    derder1_normalized=derivee1/np.sum(abs(derivee1))
    derder2_normalized=derivee2/np.sum(abs(derivee2))
    res=[]
    if max1>max2: 
        res.append(max1/max2 -1)
    else : 
        res.append(max2/max1-1)
    if maxder1>maxder2:
        res.append(maxder1/maxder2-1)
    else :
        res.append(maxder2/maxder1-1)
    if maxderder1>maxderder2:
        res.append(maxderder1/maxderder2-1)
    else :
        res.append(maxderder2/maxderder1-1)
    res.append(wasserstein_distance(ar1_normalized, ar2_normalized))
    res.append(wasserstein_distance(der1_normalized, der2_normalized))
    res.append(wasserstein_distance(derder1_normalized, derder2_normalized))
    res=np.array(res)
    return np.sum(res**2)





def dissemblance_1(pandemic1, pandemic2, pandemic3, pandemic4):
    return np.sum([abs(pandemic1[i]-pandemic2[i]) for i in range(len(pandemic1))])+np.sum([abs(pandemic1[i]-pandemic3[i]) for i in range(len(pandemic1))])+np.sum([abs(pandemic1[i]-pandemic4[i]) for i in range(len(pandemic1))])+np.sum([abs(pandemic2[i]-pandemic3[i]) for i in range(len(pandemic1))])+np.sum([abs(pandemic2[i]-pandemic4[i]) for i in range(len(pandemic1))])+np.sum([abs(pandemic3[i]-pandemic4[i]) for i in range(len(pandemic1))])


def dissemblance_2(pandemic1, pandemic2, pandemic3, pandemic4): 
    return diff_between_2_arrays(pandemic1, pandemic2)+diff_between_2_arrays(pandemic1, pandemic3)+diff_between_2_arrays(pandemic1, pandemic4)+diff_between_2_arrays(pandemic2, pandemic3)+diff_between_2_arrays(pandemic2, pandemic4)+diff_between_2_arrays(pandemic3, pandemic4)


def dissemblance_3(pandemic1, pandemic2, pandemic3, pandemic4): 
    pandemic1_normalized = np.array(pandemic1/sum(np.abs(pandemic1)))
    pandemic2_normalized = np.array(pandemic2/sum(np.abs(pandemic2)))
    pandemic3_normalized = np.array(pandemic3/sum(np.abs(pandemic3)))
    pandemic4_normalized = np.array(pandemic4/sum(np.abs(pandemic4)))
    return wasserstein_distance(pandemic1_normalized, pandemic2_normalized)+wasserstein_distance(pandemic1_normalized, pandemic3_normalized)+wasserstein_distance(pandemic1_normalized, pandemic4_normalized)+wasserstein_distance(pandemic2_normalized, pandemic3_normalized)+wasserstein_distance(pandemic2_normalized, pandemic4_normalized)+wasserstein_distance(pandemic3_normalized, pandemic4_normalized)


def dissemblance_4(pandemic1, pandemic2, pandemic3, pandemic4): 
    return diff_between_2_arrays_2(pandemic1, pandemic2)+diff_between_2_arrays_2(pandemic1, pandemic3)+diff_between_2_arrays_2(pandemic1, pandemic4)+diff_between_2_arrays_2(pandemic2, pandemic3)+diff_between_2_arrays_2(pandemic2, pandemic4)+diff_between_2_arrays_2(pandemic3, pandemic4)

