import numpy as np



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
