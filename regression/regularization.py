# This example shows different regularization approaches to reduce overfitting
# We use a modification and curated version of the Housing Prices Dataset from
# https://www.kaggle.com/datasets/yasserh/housing-prices-dataset
# stored in (data/housing_curated.csv).

from sklearn.linear_model import SGDRegressor
import utils
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
import pandas as pd

pd.set_option('display.max_columns', None)  # Display all columns of a DataFrame in pandas
random_state = 42

# LOAD THE CURATED DATASET
dataset_file_name = 'data/housing_curated.csv'
integer_independent_vars = ['bedrooms', 'bathrooms', 'stories', 'parking', 'width', 'length', 'quality']
binary_independent_vars = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea',
            'furnishingstatus_furnished','furnishingstatus_semi-furnished','furnishingstatus_unfurnished']
independent_vars = integer_independent_vars + binary_independent_vars
dependent_var = 'price'

# Split the dataset into training and testing sets
(X_train, y_train), (X_test, y_test) = utils.load_dataset_from_csv(dataset_file_name, independent_vars,
                                                                   dependent_var, 0.2, random_state)
print(f"Training set shape: {X_train.shape}.")
print("Training set description:\n", X_train.describe(), end='\n\n')

# Scale the dataset
X_train_scaled, X_test_scaled = utils.scale_X_dataset(RobustScaler(), X_train, X_test)

print("Model performance with the original features:")
utils.show_regression_performance(
    *utils.create_SDG_regression_model_and_evaluate(X_train_scaled, y_train, X_test_scaled, y_test, random_state))

# CREATE MULTIPLE POLYNOMIAL FEATURES

# Create the polynomial features with degree of 3
poly_features = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

print(f"Training set shape: {X_train_poly.shape}.")

# Scale the dataset
X_train_scaled, X_test_scaled = utils.scale_X_dataset(RobustScaler(), X_train_poly, X_test_poly)

print("Model performance after adding the polynomial features:")
utils.show_regression_performance(
    *utils.create_SDG_regression_model_and_evaluate(X_train_scaled, y_train, X_test_scaled, y_test, random_state))

# QUESTION: Does the model's performance improve after adding polynomial features?
# ANSWER: No, it deacreases.
# QUESTION: Why?
# ANSWER: The model is overfitting the training data. The are are two many features (and the complexity of the model increases)
# leading to overfitting.
# QUESTION: How can we reduce overfitting?
# ANSWER: One option is using feature selection.


# FEATURE SELECTION
# Let's select the most important features using the SelectKBest
features_to_select = len(independent_vars)
selector = SelectKBest(score_func=f_regression, k=features_to_select)  # Select same number as the original independent vars
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

print(f"Model performance after adding the polynomial features and selecting the best {features_to_select} features:")
utils.show_regression_performance(
    *utils.create_SDG_regression_model_and_evaluate(X_train_selected, y_train, X_test_selected, y_test, random_state))

# QUESTION: Does the model's performance improve after selecting the best features? Why?
# ANSWER: Yes, because the model is less complex and overfitting is reduced.
# QUESTION: Is the performance better than the original model? Why?
# ANSWER: Yes, because some derived features are more relevant that some original ones.
# QUESTION: What would you think it would happen if we halve the number of selected features?
# ANSWER: The performance would decrease, as we would be removing relevant features, making the model too simple (underfitting).
# Do it!


# HARD FEATURE SELECTION
# Let's select the most important features using the SelectKBest
features_to_select = len(independent_vars)//2
selector = SelectKBest(score_func=f_regression, k=features_to_select)  # Select same number as the original independent vars
X_train_half_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_half_selected = selector.transform(X_test_scaled)

print(f"Model performance after adding the polynomial features and selecting the best {features_to_select} features:")
utils.show_regression_performance(
    *utils.create_SDG_regression_model_and_evaluate(X_train_half_selected, y_train, X_test_half_selected, y_test, random_state))

# Exactly, the model's performance decreases because we are removing relevant features, making the model too simple (underfitting).
# QUESTION: How do you think we could find the optimal number of features to select?
# ANSWER: Using a validation set, we could try different values, estimate the best one and peform the final
# evaluation of the model's performance using the test set.

# L1 REGULARIZATION (LASSO REGRESSION)
# Let's apply L1 regularization to polynomial features with all the features
# Create a SDG regression with L1 regularization
# We first select too many features on purpose to see the effect of L1 regularization
features_to_select = 5*len(independent_vars)
alpha = 5e3
selector = SelectKBest(score_func=f_regression, k=features_to_select)  # Select same number as the original independent vars
X_train_many_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_many_selected = selector.transform(X_test_scaled)
model = SGDRegressor(penalty='l1', alpha=alpha, random_state=random_state)
model.fit(X_train_many_selected, y_train)
y_pred = model.predict(X_test_many_selected)
utils.plot_actual_vs_predicted_values(y_test, y_pred)
# Model evaluation
print(f"Model performance after adding the polynomial features and L1 regularization:")
metrics = utils.evaluate_regression(y_test, y_pred)
utils.show_regression_performance(*metrics, end='\n')
print(f"Out of {features_to_select} to select, L1 (alpha={alpha}) has reduced them to {len(model.coef_.nonzero()[0])}.\n")

# QUESTION: What is the effect of L1 regularization?
# ANSWER: L1 regularization reduces the number of selected features to the most relevant ones (feature selection).
#         Particularly, it reduces 80 features to 57.
# QUESTION: What happens if we increase the alpha parameter (let's say to 1e50)?
# ANSWER: The model will be simpler, but it may underfit the data, dropping the performance.
# Try it!
# QUESTION: What happens if we decrease the alpha parameter (let's say to 1e-5)?
# ANSWER: The model will be more complex, but it may overfit the data, dropping the performance.
# Try it!


# L1 + L2 REGULARIZATION (LASSO + RIDGE REGRESSION) = ELASTIC NET REGRESSION
# By default, SGDRegressor uses L2 regularization.
# L2 regularization prevents coefficients from becoming too large, but it does not reduce them to zero.
# Elastic Net combines L1 and L2 regularization
# Thus, it can reduce the number of features and prevent coefficients from becoming too large.
# Let's apply Elastic Net to the polynomial features with all the features

features_to_select = 5*len(independent_vars)
alpha = 1
selector = SelectKBest(score_func=f_regression, k=features_to_select)  # Select same number as the original independent vars
X_train_many_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_many_selected = selector.transform(X_test_scaled)
model = SGDRegressor(penalty='elasticnet', alpha=alpha, l1_ratio=0.6, random_state=random_state)
model.fit(X_train_many_selected, y_train)
y_pred = model.predict(X_test_many_selected)
utils.plot_actual_vs_predicted_values(y_test, y_pred)
# Model evaluation
print(f"Model performance after adding the polynomial features and L1+L2 regularization:")
metrics = utils.evaluate_regression(y_test, y_pred)
utils.show_regression_performance(*metrics, end='\n')
print(f"Out of {features_to_select} to select, L1+L2 (alpha={alpha}) has reduced "
      f"them to {len(model.coef_.nonzero()[0])}.\n")

# QUESTION: Has the L1+L2 regularization improved the model's performance? Why?
# ANSWER: Yes. In this particular case scenario, with L2 it has not been necessary
# to drop any feature and the model has improved.
