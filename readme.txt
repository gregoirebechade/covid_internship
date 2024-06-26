This code enables to use and compare different forecast of pandemics propagations. 

The ipynb notebooks correspond to the different step of implementation of the models. 
The models are in the \models directory. 

* Notebook 00 shows how the first pandemics were generated and how the 324 diverse pandemics used of model evaluation were generated. 
* Notebook 01 implements the SIR model with Euler method
* Notebook 02 implements an ARIMA model and first fits to the data 
* Notebook 03 is used to create the loss for evaluating the performances of the models. 
* Notebook 04 is used to make some teste, it does not contain anything useful. 
* Notebook 05 is used to implement the exponential regressions models. 
* Notebook 06 is used to implement the moving average model. 
* Notebook 07 is used to make a global evaluation of the models on a single pandemic. 
* Notebook 08 is used to adapt the previois code to multi dimensional models (i.e models that take into account mobility and infectious data). 
* Notebook 09 is used to implement the SIRH model, a variation of the SIR model. 
* Notebook 10 is used to implement machine learning models : linear and bayesian regressions; 
* Notebook 11 contains the figures in the paper. 
* Notebook 12 is used to analyze the global evaluation on all 324 pandemics generated. 

The other files at the root of the project are : 

- comparison_of_models_n_hospitalized.py and comparison_of_models.py, which enable to compare the models on a single pandemic, with new_deaths or n_hospitalized as the target. 
- the five .csv file contain the data of the first pandemics that were used for the first tests. 
- the generating_pandemics.py file was used to generate the 324 pandemics with Covasim. 
- the global_comparison.py was used to test all the models on the 324 pandemics generated. 
- the Mcmc.py file was used to perform a Monte Carlo Markov Chain to search for optimal parameters. 


The \results directory contains : 
    - global_evaluation_from_zero and global_evaluation_from_zero_corrected, which contains ths json file of the performance of the models and the performance of the models without inf values. 
    - predictions_of_the_models and predictions_of_the_models, which contains csv file of the prediction of the models and corrected csv_files without the inf values. 
    - other .json or .txt files that correspond to outputs of the notebooks for simple tests. 


The \all_pandemics directory contains 324 csv file that correspond to the 324 pandemics generated with the number of infected, the number of hospitalized, the mobility used and the value of the reproduction number (R_eff). See the generating_pandemics.py file for more details on the generation of the pandemics. 

The \models directory contains the .py file of the different models : 
    - Model.py is the base class of all the models and contains all useful functions that a model should have. 
    - useful_functions.py contains many useful functions accessed in the code. 
    - evaluate_model.py contains the implementation of the functions used to assess the performance of the models. 
    - Arima.py contains the implementation of both VAR and ARIMA models. 
    - BayesianRegression contains the implementation of the Bayesian regression model. 
    - exponential_regression.py contains the implementation of both exponential regression and exponential regression multi models. 
    - LinearRegression.py contains the implement of the linear regression model. 
    - moving_average.py model contains the implementation of the moving average model.
    - SIRD.py contains the implementation of the SIRD model, a variation of the SIR model with a D ( for deaths ) compartment. 
    - SIRH.py contains the implementation of the SIRH model, a variation of the SIR model with a H ( for hospitalized ) compartment. 
