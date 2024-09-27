from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import seaborn as sns

def plot_values(X_set: pd.DataFrame, y_set: pd.Series, x_label: str, y_label: str) -> None:
    """Plot the values of the training set"""
    plt.scatter(X_set, y_set, color='blue', marker='o', alpha=0.5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'Scatter plot of "{x_label}" and "{y_label}"')
    plt.show()


def plot_actual_vs_predicted_values(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    """Plot actual vs predicted values"""
    plt.scatter(y_test, y_pred, color='blue', marker='o', alpha=0.5)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    # Add the red line y=x
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linewidth=2)
    plt.grid(True)
    plt.show()


def load_dataset_from_csv(csv_file: str, independent_vars: list, dependent_var: str, test_size: float,
                          random_state: int)\
        -> Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]:
    """ Load a dataset from a CSV file, select the independent and dependent variables, and split the dataset into
       training and testing sets."""
    # Load the dataset from the CSV file
    dataset = pd.read_csv(csv_file)
    # Select the independent and dependent variables
    X = dataset[independent_vars]
    y = dataset[dependent_var]
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return (X_train, y_train), (X_test, y_test)


def visualize_correlation(independent_vars: pd.DataFrame, dependent_var: pd.DataFrame, n_plot_columns: int) -> None:
    """Visualize the correlation between the independent variables and the dependent variable."""
    n_columns_df = independent_vars.shape[1]
    n_plot_rows = n_columns_df // n_plot_columns if n_columns_df % n_plot_columns == 0 else (n_columns_df // n_plot_columns) + 1
    fig, axs = plt.subplots(n_plot_rows, n_plot_columns, figsize=(n_plot_columns*5, 5))
    # Loop over each column in independent_vars and create a scatter plot in a subplot
    for i, column in enumerate(independent_vars.columns):
        axs[i].scatter(independent_vars[column], dependent_var)
        axs[i].set_xlabel(column)
        axs[i].set_ylabel(dependent_var.name)
        axs[i].set_title(f'{dependent_var.name} vs {column}')
    plt.tight_layout()  # Adjust the layout to prevent overlapping
    plt.show()


def visualize_influence(independent_vars: pd.DataFrame, dependent_var: pd.DataFrame) -> None:
    """Visualize one grid figure where each plot shows the influence of the independent variable on the dependent variable."""
    n_features = independent_vars.shape[1]
    grid_size = int(np.ceil(np.sqrt(n_features)))  # Calculate grid size
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size*3, grid_size*3))
    # Flatten the axes array for easy iteration
    axs = axs.flatten()
    # Loop over each column in independent_vars and create a scatter plot in a subplot
    for i, column in enumerate(independent_vars.columns):
        axs[i].scatter(independent_vars[column], dependent_var)
        axs[i].set_xlabel(column)
        axs[i].set_ylabel(dependent_var.name)
        axs[i].set_title(f'{dependent_var.name} vs {column}')
    # Remove unused subplots
    for j in range(n_features, grid_size*grid_size):
        fig.delaxes(axs[j])
    plt.tight_layout()  # Adjust the layout to prevent overlapping
    plt.show()


def describe_array(array: np.array) -> None:
    """Describe the contents of a NumPy array."""
    assert array.ndim == 2, "The array must have 2 dimensions."
    print("Overall Shape:", array.shape)
    print("Overall Size:", array.size)
    print("Overall Data type:", array.dtype)
    print()
    # Describe each column
    for i in range(array.shape[1]):
        print(f"Column {i + 1} Statistics:")
        print("  Mean:", np.mean(array[:, i]))
        print("  Standard Deviation:", np.std(array[:, i]))
        print("  Minimum:", np.min(array[:, i]))
        print("  Maximum:", np.max(array[:, i]))
        print("  Median:", np.median(array[:, i]))
        print("  25th Percentile:", np.percentile(array[:, i], 25))
        print("  75th Percentile:", np.percentile(array[:, i], 75))
        print()



def plot_feature_value_distribution(data_frame: pd.DataFrame) -> None:
    """Plot the distribution of the features in the data frame."""
    # Identify feature types
    int_features = data_frame.select_dtypes(include=['int64', 'int32']).columns
    float_features = data_frame.select_dtypes(include=['float64', 'float32']).columns
    categorical_features = data_frame.select_dtypes(include=['object', 'category']).columns
    # Combine all features
    features_names = list(int_features) + list(float_features) + list(categorical_features)
    num_features = len(features_names)
    # Determine grid size
    num_rows = (num_features + 2) // 3  # Adjust this depending on how you want to layout your grid
    # Create the subplots
    fig, axes = plt.subplots(num_rows, 4, figsize=(20, num_rows * 2))
    axes = axes.flatten()
    # Plot each feature
    for i, feature in enumerate(features_names):
        if feature in int_features or feature in float_features:
            sns.histplot(data_frame[feature], kde=True, bins=30, ax=axes[i])
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')
        elif feature in categorical_features:
            data_frame[feature].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, cmap='viridis', ax=axes[i])
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].set_ylabel('')  # Hide the y-label
    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    # Adjust the layout and create the show the figure
    plt.tight_layout()
    plt.show()


def detect_outliers_iqr(df: pd.DataFrame, threshold: float = 1.5) -> pd.DataFrame:
    """
    This function takes a DataFrame df of features and returns a DataFrame df_outliers of the same shape
    indicating True for outliers and False for non-outliers according to the IQR method.
    """
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df_outliers = (df < (Q1 - threshold * IQR)) | (df > (Q3 + threshold * IQR))
    return df_outliers



def evaluate_regression(y_test: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float]:
    """Evaluate the regression model using the MSE, RMSE, MAE, and R-squared metrics."""
    # MSE = Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    # RMSE = Root Mean Squared Error
    rmse = np.sqrt(mse)
    # MAE = Mean Absolute Error
    mae = np.mean(np.abs(y_test - y_pred))
    # R-squared = coefficient of determination
    r2 = r2_score(y_test, y_pred)
    return mse, rmse, mae, r2


def store_dataset(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, file_name: str) -> None:
    """Store the training, validation, and test sets in a CSV file."""
    train_dataset = pd.concat([X_train, y_train], axis=1)
    test_dataset = pd.concat([X_test, y_test], axis=1)
    pd.concat([train_dataset, test_dataset], axis=0).to_csv(file_name, index=False)


def scale_X_dataset(scaler, X_train_p: pd.DataFrame, X_test_p: pd.DataFrame, integer_independent_vars: list = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Scale the integer independent variables of the dataset using the provided scaler.
    If integer_independent_vars is None, all the columns will be scaled."""
    if integer_independent_vars is None:
        return scaler.fit_transform(X_train_p), scaler.transform(X_test_p)
    else:
        return scaler.fit_transform(X_train_p[integer_independent_vars]), scaler.transform(X_test_p[integer_independent_vars])


def create_SDG_regression_model_and_evaluate(X_train_p: pd.DataFrame, y_train_p: pd.Series, X_test_p: pd.DataFrame,
                                             y_test_p: pd.Series, random_state: int) -> \
        Tuple[float, float, float, float]:
    """Create a Stochastic Gradient Descent (SGD) regression model and evaluate its performance."""
    model = SGDRegressor(random_state=random_state)
    model.fit(X_train_p, y_train_p)
    # Predict the values for the test set
    y_pred = model.predict(X_test_p)
    plot_actual_vs_predicted_values(y_test_p, y_pred)
    # Model evaluation
    mse, rmse, mae, r2 = evaluate_regression(y_test_p, y_pred)
    return mse, rmse, mae, r2


def show_regression_performance(mse: float, rmse: float, mae: float, r2: float, end='\n\n') -> None:
    """Show the performance of the regression model."""
    print(f'Mean Squared Error (MSE): {mse:.4f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
    print(f'Mean Absolute Error (MAE): {mae:.4f}')
    print(f'R-squared determination coefficient: {r2:.4f}', end=end)

