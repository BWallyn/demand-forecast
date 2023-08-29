# Demand Forecast: Project to forecast demands of newspapers

The goal of this project is to forecast the demand of newspapers. To do so, we decompose the project in several parts.

## Summary

1. Data exploration: The first part is data exploration. To do so, we analyze the different features available, the time period, the different elements.
2. ACF and PACF: Analyze the ACF and PACF by ID in order to get the possible lags to add as a feature to the dataset and use to predict.
3. Train model: Notebook to train a Gradient Boosting method using Catboost and try different hyperparameters on a cross validation using expanding windows. Log every run on MLflow to store all tests, metrics, hyperparameters and models. Finally, train a model with best hyperparameters on the whole train set.
4. Analyze model: Analyze the final model on the train and test set and compare to the solution furnished.