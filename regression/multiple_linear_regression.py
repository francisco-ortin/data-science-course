# This file shows an example of multiple linear regression using Stochastic Gradient Descent.
# We use the employee salaries dataset in https://www.kaggle.com/datasets/yasserh/employee-salaries-datatset
# stored in (data/salary.csv).
# The dataset contains the following columns:
# - Gender: 0-Female, 1-Male
# - Age: Age of the employee (years)
# - PhD: Whether the person has a PhD (0-No, 1-Yes)
# - Salary (dependent variable / target): Salary of the employee (in K USD per year)

import utils
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

random_state = 42

# LOAD THE DATASET
(X_train, y_train), (X_test, y_test) = utils.load_dataset_from_csv('data/salary.csv', ['Gender', 'Age', 'PhD'],
                                                                   'Salary', 0.2, random_state=random_state)
print("Description of the training set before scaling:")
print(X_train.describe(), end='\n\n')

# Scale X_train and X_test 'Age' feature using a Max-Min Scaler between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
X_train['Age'] = scaler.fit_transform(X_train[['Age']])
# Shows X_train after scaling
print("Description of the training set after scaling 'Age':")
print(X_train.describe(), end='\n\n')
# Important: Use the same scaler that was used for X_train, without fitting again
X_test['Age'] = scaler.transform(X_test[['Age']])

# Visualize the correlation between the features in X_train and the target variable y_train
# We can see the influence of each feature on the target variable.
utils.visualize_correlation(X_train, y_train, n_plot_columns=3)


# CREATE THE MULTIPLE LINEAR REGRESSION MODEL
# Let's use the Stochastic Gradient Descent (SGD) regressor
model = SGDRegressor(random_state=random_state)
model.fit(X_train, y_train)

# Predict the values for the test set
y_pred = model.predict(X_test)
utils.plot_actual_vs_predicted_values(y_test, y_pred)


# MODEL EVALUATION

mse, rmse, mae, r2 = utils.evaluate_regression(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'R-squared determination coefficient: {r2:.4f}', end='\n\n')


# MODEL INTERPRETATION

# Shows the intercept of the model
print(f"Intercept of the model: {model.intercept_[0]}.")
# Shows the coefficients of the model, indicating its name and the importance of each feature
coefficients = dict(zip(X_train.columns, model.coef_))
print(f"Coefficients of the model: {coefficients}.")


# Questions:
# 1) Does the model perform well?
# Answer: The model does not perform well. The predicted values are far from the actual values.
# 2) How could you answer the previous question more objectively (using a metric)?
# Answer: We could use the R-squared determination coefficient. The value is close to 0, which means that the model
#         does not explain any variability of the dependent value.
# 3) Why is not the model performing well?
# Answer: Because a linear model cannot capture the relationship between the features and the target variable.
# 4) What is the variable that influences the salary the most?
# Answer: The 'Age' variable is the most influential, as it has the highest coefficient.
# 5) Is that influence positive or negative?
# Answer: The influence is positive. Greater age implies a higher salary.
# 6) What is the influence of the rest of the variables?
# Answer: All of them influence positively. None of them are close to zero (no influence).
# PhD influences more than Gender in the Salary.


