# kaggle_housing_regression
This is my attempt at kaggle playground exercise "Advanced Housing Regression". The goal is to predict housing price, given the standard information about the house. i.e. how many bedrooms/bathrooms, square footage of living room/garage, years that it has been built.

# Feature Selection
Select the most informative features before you pass them to the regression model that you choose. We are seeking those features that has positive association with sale price.

Here is a plot that shows the relationship between numeric columns and sale price.
![floating_columns](https://cloud.githubusercontent.com/assets/1633731/22074801/9eb9a3e4-dd77-11e6-8741-1504669a7977.png)


# Normalize features.
Notice some features are much bigger number than other features. An issue arise that those feature with big values will be so important in regression models that other features don't even matter. Because other features will have little impact when we sum up ( feature value * coefficient ). 

![log_columns](https://cloud.githubusercontent.com/assets/1633731/22075268/5c9f4a0c-dd79-11e6-9f7e-5a304d08086b.png)


# Hyperparameter tuning
Sklearn provide a method to search for optimal parameter value. It is basically searching for the best performing combination of parameters. Each parameter is given a list of possible values it can take on. The tuning is based on the scoring mechanism you specify. For regression, it is using MSE. For classification, the scoring mechanism is _____ .
