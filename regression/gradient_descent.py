from statistics import LinearRegression
# We create a simple linear model using Gradient Descent (GD) as the optimizer.
# Remember, GD is an appropriate optimization algorithm to compute the parameters of a linear regression
# model when the dataset has a large number of instances.
# scikit-learn provides SGDRegressor (Stochastic Gradient Descent) for this purpose.

# We use the Statistics Online Computational Resource (SOCR) dataset for human heights (inches) and weight (pounds).
# We will try to predict the height from the weight.


from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import utils

ramdom_state = 42

# DATA PREPARATION

# Load the dataset
(X_train, y_train), (X_test, y_test) = utils.load_dataset_from_csv('data/height_weight.csv',
                            ['Height'],'Weight', 0.2, random_state=ramdom_state)
# Let's print some data
print("First values X_train:", X_train.head(10), sep='\n')
print("First values y_train:", y_train.head(10), sep='\n')

# Visualize the data
utils.plot_values(X_train, y_train, 'Height', 'Weight')

# MODEL CREATION AND EVALUATION WITH LINEAR REGRESSION

# We first create, train and evaluate a LinearRegression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the values for the test set
y_pred = model.predict(X_test)
utils.plot_actual_vs_predicted_values(y_test, y_pred)

# Shows MSE and R-square for the test set, and the model parameters
mse = mean_squared_error(y_test, y_pred)
print(f'LinearRegression: Mean Squared Error (MSE) for the test set: {mse:.4f}')
print(f"LinearRegression: R-squared determination coefficient: {r2_score(y_test, y_pred):.4f}.")


# MODEL CREATION AND EVALUATION WITH (STOCHASTIC) GRADIENT DESCENT REGRESSION

# Create and train the LinearRegression model
model = SGDRegressor(random_state=ramdom_state)
model.fit(X_train, y_train)

# Predict the values for the test set
y_pred = model.predict(X_test)
utils.plot_actual_vs_predicted_values(y_test, y_pred)

# Shows MSE and R-square for the test set, and the model parameters
mse = mean_squared_error(y_test, y_pred)
print(f'SGDRegression: Mean Squared Error (MSE) for the test set: {mse:.4f}')
print(f"SGDRegression: R-squared determination coefficient: {r2_score(y_test, y_pred):.4f}.")

# Questions:
# 1) Why do you think SGD is behaving much worse than OLS with the same data? Try to fix it.
# This is an important point that occurs in most neural networks.
# Answer: The data is not scaled. We need to scale the data before using GD.
#         GD is very sensitive to the scale of the data. The reason is that the learning rate is the same for all
#         features, and if the features have different scales, the learning rate will be too high for some features
#         and too low for others. This will make the algorithm converge very slowly or not converge at all.
#         We can fix it by scaling the data using the StandardScaler.

# ------------- SOLUTION ------------------

# Scale X_train and X_test using the StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Important: Use the same scaler that was used for X_train, without fitting again
X_test = scaler.transform(X_test)

# THE REST OF THE CODE REMAINS THE SAME
model = SGDRegressor(random_state=ramdom_state)
model.fit(X_train, y_train)

# Predict the values for the test set
y_pred = model.predict(X_test)
utils.plot_actual_vs_predicted_values(y_test, y_pred)

# Shows MSE and R-square for the test set, and the model parameters
mse = mean_squared_error(y_test, y_pred)
print(f'SGDRegression: Mean Squared Error (MSE) for the test set: {mse:.4f}')
print(f"SGDRegression: R-squared determination coefficient: {r2_score(y_test, y_pred):.4f}.")


