# kaggle_housing_regression
This is my attempt at kaggle playground exercise "Advanced Housing Regression". The goal is to predict housing price, given the standard information about the house. i.e. how many bedrooms/bathrooms, square footage of living room/garage, years that it has been built.

# Feature Selection
Select the most informative features before you pass them to the regression model that you choose. We are seeking those features that has positive association with sale price.

Here is a plot that shows the relationship between columns that contain numeric values and sale price.
![floating_columns](https://cloud.githubusercontent.com/assets/1633731/22074801/9eb9a3e4-dd77-11e6-8741-1504669a7977.png)


# Normalize features.
Since we are regressing on many features and some features are much bigger number than the rest, those bigger features need to normalized while preserving the variances so that they wouldn't dominate other features. We want to give every feature equal opportunity to contribute to the model.

# Hyperparameter tuning
Sklearn provide a method to search for optimal parameter value. It is basically searching for the best performing combination of parameters. Each parameter is given a list of possible values it can take on. The tuning is based on the scoring mechanism you specify. For regression, it is using MSE. For classification, the scoring mechanism is _____ .
