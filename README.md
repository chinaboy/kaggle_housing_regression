# kaggle_housing_regression
This is my attempt at kaggle playground exercise "Advanced Housing Regression". The goal is to predict housing price, given the standard information about the house. i.e. how many bedrooms/bathrooms, square footage of living room/garage, years that it has been built.

# Feature Selection
Select the most informative features before you pass them to the regression model that you choose. We are seeking those features that has positive association with sale price.

Here is a plot that shows the relationship between numeric columns and sale price.
![floating_columns](https://cloud.githubusercontent.com/assets/1633731/22074801/9eb9a3e4-dd77-11e6-8741-1504669a7977.png)


# Normalize Numeric features.
Notice some features are much bigger number than other features. An issue arise that those feature with big values will be so important in regression models that other features don't even matter. Because other features will have little impact when we sum up ( feature value * coefficient ). Here log() function is used to scale down big numeric value. Here is the plot of columns after they've been transformed. And all columns will be scaled by a StandardScaler provided by sklearn.

![log_columns](https://cloud.githubusercontent.com/assets/1633731/22075268/5c9f4a0c-dd79-11e6-9f7e-5a304d08086b.png)

# Add dummy variable for categorical features
Categorical features could be helpful. So convert them into dummy variables.

# Backward Feature selection
Backward feature selection is a procedure to weed out those features in a greed way. I found this following pseudo code explain it the best.

Loop through n features. 
    For each feature, 
       exclude itself from the rest to generate a new set of features.
       Search for best parameter. 
       Fit a regression model with found param.
       Calculate the score.
       Record those score.
    Find the best performing features set S'.
    The least effective feature is the one discarded out of S'.
    Return a tuple consisting of the discarded feature and the best performing feature set.   


# Hyperparameter tuning
Sklearn provide a method GridSearch() to search for optimal parameter value. It tests all combinations of parameters. Each parameter is given a list of possible values it can take on. The tuning is based on the scoring mechanism you specify. For regression, it is using MSE.
