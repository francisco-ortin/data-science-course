# Let's create a simple linear model using scikit-learn.
from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

random_state = 42
# domain of the function
from_value, to_value = -10, 10

# Generate the dataset from a secret polynomial function
def generate_dataset(n_samples: int, random_state: int, from_value: float, to_value: float) -> Tuple[pd.DataFrame, np.array]:
    """Generate a dataset with n_samples samples for a secret polynomial function."""
    np.random.seed(random_state)
    x_values = np.linspace(from_value, to_value, n_samples)
    # Ground truth polynomial function of degree 3
    y_truth = 3.1 * x_values ** 3 + 2.2 * x_values ** 2 + 1.1 * x_values + 5.5
    # Add Gaussian noise
    y_values = y_truth + np.random.normal(0, 100, n_samples)
    return pd.DataFrame({'x': x_values, 'y': y_values}), y_truth


dataset, y_truth = generate_dataset(1000, random_state, from_value, to_value)

# Split the dataset into train, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(dataset[['x']], dataset['y'], test_size=0.2,
                                                            random_state=random_state)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25,
                                                  random_state=random_state)  # 0.25 * 0.8 = 0.2


def plot_model(model, degree, X_train_p, y_train_p, y_truth_p, from_value, to_value):
    """Function to plot a polynomial regression model."""
    plt.figure(figsize=(14, 8))
    plt.scatter(X_train_p, y_train_p, color='gray', alpha=0.5, label='Training data')
    plt.plot(dataset['x'], y_truth_p, color='red', label='Ground truth (degree 3)')
    X_plot = np.linspace(from_value, to_value, 100).reshape(-1, 1)
    X_plot_poly = poly_features.transform(X_plot)
    y_plot_pred = model.predict(X_plot_poly)
    plt.plot(X_plot, y_plot_pred, linestyle='--', label=f'Polynomial degree {degree}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'Polynomial Regression Model (degree {degree})')
    plt.legend()
    plt.show()


# Train polynomial regression models with different degrees
degrees = list(range(1, 11)) + [20, 50, 100]
best_degree = 0
best_mse = float('inf')
best_model = None

for degree in degrees:
    # Create the polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(X_train.values)
    X_val_poly = poly_features.transform(X_val.values)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # Evaluate the model on the validation set
    y_val_pred = model.predict(X_val_poly)
    mse = mean_squared_error(y_val, y_val_pred)

    print(f'Degree: {degree:3}, Validation MSE: {mse:12.2f}')

    plot_model(model, degree, X_train, y_train, y_truth, from_value, to_value)

    if mse < best_mse:
        best_mse = mse
        best_degree = degree
        best_model = model

# Evaluate the best model on the test set
poly_features_best = PolynomialFeatures(degree=best_degree)
X_test_poly = poly_features_best.fit_transform(X_test.values)
y_test_pred = best_model.predict(X_test_poly)
test_mse = mean_squared_error(y_test, y_test_pred)

print(f'Best degree: {best_degree}.')
print(f'Test MSE: {test_mse:.4f}.')


def plot_results(X_train, y_train, y_truth, X_test, y_test, best_model, best_degree):
    """Plot the results of the polynomial regression model."""
    plt.figure(figsize=(14, 8))
    plt.scatter(X_train, y_train, color='gray', alpha=0.5, label='Training data')
    plt.plot(dataset['x'], y_truth, color='blue', label='Ground truth (degree 3)')
    plt.scatter(X_test, y_test, color='red', alpha=0.7, label='Test data')

    # Plot the best model's predictions
    X_plot = np.linspace(-10, 10, 100).reshape(-1, 1)
    X_plot_poly = poly_features_best.transform(X_plot)
    y_plot_pred = best_model.predict(X_plot_poly)
    plt.plot(X_plot, y_plot_pred, color='green', linestyle='--', label=f'Best model (degree {best_degree})')

    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Polynomial Regression Model Evaluation')
    plt.legend()
    plt.show()

plot_results(X_train, y_train, y_truth, X_test, y_test, best_model, best_degree)


# Questions:
# 1) What is the best degree for the polynomial regression model?
# Answer: The best degree for the polynomial regression model is 4.
# 2) Does the degree coincide with the degree of the ground truth polynomial function? Why?
# Answer: No, it does not. The reason is that more complexity might slightly fit better the data.
#         A big difference might produce overfitting, but not a slight difference (depends on the data and its noise).
# 3) What happens when for degrees 1 and 2?
# Answer: For degrees 1 and 2, the model is too simple and underfits the data (high MSE).
# 4) What happens when for degrees 50, and 100?
# Answer: The model is too complex and overfits the data (high MSE).
# 5) Is the test MSE higher or lower than the validation MSE? Why?
# Answer: The test MSE is higher than the validation MSE because the model was trained to minimize the validation MSE.
# 6) What do you learn from this last question?
# Answer: The validation MSE is an optimistic estimate of the model's performance.
#         Remember, the validation set should not be used to report the performance!!!!
