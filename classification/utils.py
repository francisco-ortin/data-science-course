from typing import Tuple, List, TypeVar, Type, Optional

import pandas as pd
from scipy import stats
from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from itertools import combinations
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, DecisionTreeClassifier


def split_dataset(dataset: pd.DataFrame, independent_vars: list, dependent_var: str, test_size: float,
                  random_state: Optional[int]) \
        -> Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]:
    """ Select the independent and dependent variables, and split the dataset into
       training and testing sets."""
    # Select the independent and dependent variables
    X = dataset[independent_vars]
    y = dataset[dependent_var]
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return (X_train, y_train), (X_test, y_test)


def show_target_per_feature_pair(X_train: pd.DataFrame, y_train: pd.Series, independent_vars: List[str],
                                 label_true_target: str, label_false_target: str) -> None:
    """Show the target variable per pairs of independent variables"""
    # Calculate the number of plots needed
    num_vars = len(independent_vars)
    num_plots = num_vars * (num_vars - 1) // 2
    # Determine the grid size
    grid_size = int(np.ceil(np.sqrt(num_plots)))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    # Iterate through each pair of independent variables and plot
    pair_idx = 0
    for var1, var2 in combinations(independent_vars, 2):
        ax = axs[pair_idx // grid_size, pair_idx % grid_size]
        # Filter points by the dependent variable and plot
        ax.scatter(X_train[var1][y_train == 1], X_train[var2][y_train == 1], color='red',
                   label=label_true_target, alpha=0.5)
        ax.scatter(X_train[var1][y_train == 0], X_train[var2][y_train == 0], color='green',
                   label=label_false_target, alpha=0.5)
        ax.set_xlabel(var1)
        ax.set_ylabel(var2)
        ax.legend()
        pair_idx += 1
    # Hide any unused subplots
    for i in range(pair_idx, grid_size ** 2):
        fig.delaxes(axs.flatten()[i])
    # Show
    plt.show()


T = TypeVar('T')  # To use generics in the following function
def scale_X_dataset(X_train: pd.DataFrame, X_test: pd.DataFrame, scaler_class: Type[T]) -> \
        Tuple[pd.DataFrame, pd.DataFrame]:
    """Scale the X_train and X_test datasets using the scaler_class"""
    # Scale X_train and X_test using the scaler_class
    scaler = scaler_class()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Convert the scaled datasets to DataFrames keeping the index (the position)
    return (pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index),
            pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index))


def plot_2d_logistic_regression_proba_classifier(X_dataset: pd.DataFrame, y_pred_proba: np.ndarray, first_feature: str,
                                                 second_feature: str, model: LogisticRegression, figure_caption: str,
                                                 probability_caption: str) -> None:
    # Extract the two features for plotting
    feature1 = X_dataset.iloc[:, 0]
    feature2 = X_dataset.iloc[:, 1]
    # Create the plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(feature1, feature2, c=y_pred_proba, cmap=plt.cm.RdYlGn_r)
    # Add a colorbar
    plt.colorbar(scatter, label=probability_caption)
    # Label the axes
    plt.xlabel(first_feature)  # First feature name
    plt.ylabel(second_feature)  # Second feature name
    plt.title(figure_caption)
    # Plot the decision boundary
    # Extract model coefficients and intercept
    w = model.coef_[0]
    b = model.intercept_[0]
    # Calculate x2 values for the decision boundary line
    x1_values = np.linspace(feature1.min(), feature1.max(), 200)
    x2_values = -(w[0] * x1_values + b) / w[1]
    plt.plot(x1_values, x2_values, color="blue", label="Decision Boundary")
    plt.legend()
    # Show the plot
    plt.show()


def compute_metrics(models, X_test: pd.DataFrame, y_test: pd.Series, metric_function) -> List[float]:
    """Compute the metric_function metric of each model"""
    # Predict the y_test values for each model
    y_predictions = [model.predict(X_test) for model in models]
    # Compute the metric given y_test and y_pred
    return [metric_function(y_test, y_pred) for y_pred in y_predictions]


def compute_AUCs(models, X_test: pd.DataFrame, y_test: pd.Series) -> List[float]:
    """Compute the AUC metric of each model"""
    # Predict the probability of y_test values for each model
    y_predictions_proba = [model.predict_proba(X_test) for model in models]
    # Compute the AUC given y_test and y_pred
    return [roc_auc_score(y_test, y_pred_proba) for y_pred_proba in y_predictions_proba]


def show_metrics(metric: str, accuracies: List[float], models) -> None:
    """Show the metric values for each model"""
    assert len(accuracies) == len(models), "The number of accuracies and models must be the same."
    print(f"Metric {metric}:")
    for model, accuracy in zip(models, accuracies):
        print(f"\t{model}: {accuracy:.4f}.")


def visualize_decision_tree(model: DecisionTreeClassifier, independent_vars: List[str],
                            class_names: List[str]) -> None:
    """Visualize a decision tree model"""
    # Visualize the decision tree
    plt.figure(figsize=(15, 10))
    plot_tree(model, feature_names=independent_vars, class_names=class_names, filled=True)
    plt.show()


def load_clean_titanic_dataset(dataset_file_name: str, independent_vars: List[str], dependent_var: str) \
        -> Tuple[pd.DataFrame, List[str]]:
    """Load the Titanic dataset, clean it and return the dataset and the list of new independent variables
    after one-hot encoding."""
    # Load the dataset
    original_dataset = pd.read_csv(dataset_file_name)
    # Filter out the features that are not going to be used
    dataset = original_dataset[independent_vars + [dependent_var]]
    # Convert the dataset so that, for the Sex feature, 'male' = 1 and 'female' = 0
    pd.options.mode.chained_assignment = None  # to avoid the following false warning
    dataset['Sex'] = dataset['Sex'].map({'male': 1, 'female': 0})
    # Convert the 'Embarked' feature to one-hot encoding
    dataset = pd.get_dummies(dataset, columns=['Embarked'])
    # return the new independent variables from the dataset
    return dataset, dataset.columns[dataset.columns != dependent_var]


def evaluate_models(models, X_test: pd.DataFrame, y_test: pd.Series, verbose: bool = True) -> List[Tuple[float,float]]:
    """Evaluate the performance of the models."""
    metrics = []
    for model in models:
        y_pred = model.predict(X_test)
        accuracy_value = accuracy_score(y_test, y_pred)
        f1_score_value = f1_score(y_test, y_pred)
        if verbose:
            print(f"Model: {model.__class__.__name__}.\n\tAccuracy: {accuracy_value:.4f}.\n\tF1 Score: {f1_score_value:.4f}.")
        metrics.append((accuracy_value, f1_score_value))
    return metrics


def confidence_interval(data: List[float], confidence_level: float) -> Tuple[float, Tuple[float, float]]:
    """Compute the confidence interval of the data with the given confidence level."""
    mean = np.mean(data)
    # Calculate the 95% confidence interval
    ci = stats.t.interval(
        confidence_level,  # Confidence level
        len(data) - 1,  # Degrees of freedom
        loc=mean,  # Mean of the data
        scale=stats.sem(data)  # Standard error of the mean
    )
    return mean, ci

