# This example shows the influence of the learning rate on the training of a model using
# the Stochastic Gradient Descent.

from typing import Tuple, List, Dict

import pandas as pd
import utils
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

random_state = 42

# Load the dataset
(X_train, y_train), (X_test, y_test) = utils.load_dataset_from_csv('data/height_weight.csv',
                      ['Height'], 'Weight', 0.2, random_state=random_state)

# Scale X_train and X_test using the StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def compute_mses_for_learning_rates(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,
                                    learning_rates: List[float], epochs: List[int]) -> Dict[int, Dict[float, float]]:
    """Compute the Mean Squared Error (MSE) for different learning rates and epochs using the SGDRegressor model."""
    # Initialize a list to store the MSE values
    mse_values_per_n_epochs = dict()
    # For each learning rate
    for epoch in epochs:
        mse_values_per_n_epochs[epoch] = dict()
        for learning_rate in learning_rates:
            # Create and train the SGDRegressor model
            model = SGDRegressor(eta0=learning_rate, max_iter=epoch, random_state=random_state)
            model.fit(X_train, y_train)
            # Predict the values for the test set
            y_pred = model.predict(X_test)
            # Compute the Mean Squared Error (MSE) for the test set
            mse = mean_squared_error(y_test, y_pred)
            mse_values_per_n_epochs[epoch][learning_rate] = mse
    return mse_values_per_n_epochs


# Create and train SDGRegressor models for different learning rates and epochs
learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0, 10]
epochs = [1, 10, 100, 1000]
mse_values = compute_mses_for_learning_rates(X_train, y_train, X_test, y_test, learning_rates, epochs)

# Show the MSE values for each learning rate and epoch
table = pd.DataFrame(mse_values)
print(f"Minimum MSE value: {table.min().min():.6f}.")  # The first min() gets the minimum value of each column,
                                                       # the second min() gets the minimum value of the resulting Series.
print("Mean Squared Error (MSE) for different learning rates and epochs:")
print(pd.DataFrame(mse_values))

# Questions:
# The following questions are very important to understand the behavior of the learning rate and training with GD
# and, in general, of neural networks.
# 1) What happens when the learning rate is too small?
# Answer: The model will take a long time to converge to the optimal solution, but it will be more accurate.
# 2) What happens when the learning rate is too large?
# Answer: The model will converge faster, but it may overshoot the optimal solution and diverge.
# 3) In general, what happens when the number of epochs is too small?
# Answer: The model will not have enough time to converge to the optimal solution.


