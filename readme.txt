# Pandemic Forecasting Models

This repository enables the use and comparison of different forecasts for pandemic propagation.

## Directory Structure

- `notebooks/`
    - `00_generate_pandemics.ipynb`: Demonstrates how the initial pandemics were generated and details the process for generating the 324 diverse pandemics used for model evaluation.
    - `01_SIR_model_Euler.ipynb`: Implements the SIR model using the Euler method.
    - `02_ARIMA_model.ipynb`: Implements the ARIMA model and fits it to the data.
    - `03_create_loss.ipynb`: Creates the loss function for evaluating model performance.
    - `04_tests.ipynb`: Contains tests and is not useful for the main implementation.
    - `05_exponential_regression.ipynb`: Implements exponential regression models.
    - `06_moving_average_model.ipynb`: Implements the moving average model.
    - `07_single_pandemic_evaluation.ipynb`: Evaluates the models on a single pandemic.
    - `08_multidimensional_models.ipynb`: Adapts previous code for multidimensional models (i.e., models that take into account mobility and infection data).
    - `09_SIRH_model.ipynb`: Implements the SIRH model, a variation of the SIR model.
    - `10_machine_learning_models.ipynb`: Implements machine learning models: linear and Bayesian regressions.
    - `11_figures.ipynb`: Contains the figures used in the paper.
    - `12_global_evaluation.ipynb`: Analyzes the global evaluation on all 324 generated pandemics.

- `models/`
    - `Model.py`: The base class for all models containing essential functions.
    - `useful_functions.py`: Contains various utility functions used throughout the code.
    - `evaluate_model.py`: Implements functions used to assess model performance.
    - `Arima.py`: Implements both VAR and ARIMA models.
    - `BayesianRegression.py`: Implements the Bayesian regression model.
    - `exponential_regression.py`: Implements both exponential regression and exponential regression multi models.
    - `LinearRegression.py`: Implements the linear regression model.
    - `moving_average.py`: Implements the moving average model.
    - `SIRD.py`: Implements the SIRD model, a variation of the SIR model with a 'D' (for deaths) compartment.
    - `SIRH.py`: Implements the SIRH model, a variation of the SIR model with an 'H' (for hospitalized) compartment.

- `results/`
    - `global_evaluation_from_zero/` and `global_evaluation_from_zero_corrected/`: Contain JSON files of model performances, including corrected versions without infinity values.
    - `predictions_of_the_models/` and `predictions_of_the_models_corrected/`: Contain CSV files of model predictions and corrected versions without infinity values.
    - Other JSON and TXT files corresponding to outputs of the notebooks for specific tests.

- `all_pandemics/`
    - Contains 324 CSV files corresponding to the 324 generated pandemics. Each file includes the number of infected, number of hospitalized, mobility used, and the reproduction number (R_eff). Refer to `generating_pandemics.py` for more details on the pandemic generation.

- Root directory files:
    - `comparison_of_models_n_hospitalized.py` and `comparison_of_models.py`: Enable comparison of models on a single pandemic, targeting new deaths or number of hospitalized individuals.
    - `first_pandemics/`: Contains five CSV files with data from the initial pandemics used for testing.
    - `generating_pandemics.py`: Generates the 324 pandemics using Covasim.
    - `global_comparison.py`: Tests all models on the 324 generated pandemics.
    - `Mcmc.py`: Performs a Monte Carlo Markov Chain search for optimal parameters.



