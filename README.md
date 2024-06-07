# Health Outcome Time Use Regression

This repository contains files to train and test regression models to map daily time spent sleeping, sedentary and active to health across 5 outcomes: stress level, resting heart rate, systolic blood pressure, distolic blood pressure and body mass index.

## Trained models

`models` contains the trained regression models:
- Version 1.0 - direct regression of time use variables to health outcomes with intercept
- Version 1.1 - direct regression, no intercept
- Version 2.0 - regression of log-ratio transformed time use variables

Each of these models can be tested with different inputs using the `model_testing.py` file

## Training

`direct_regression/training.py` can be used for training the direct regression modes, and `ilr_regression/training.py` can be used to train the log-ratio regression models. Both these files also calculate and test the accuracy of the models after training. 

## Data formatting

The [this dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset) is utilised in this repository to train the regression models. `data_formatting.py` contains the required formatting to be done on this dataset before it can be used to train the regression models. `ilr_regression/ilr_data.py` contains additionally formatting to calculate the log-ratio variables.