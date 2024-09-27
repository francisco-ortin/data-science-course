# This example creates a multiple regression model
# We use a modification of the Housing Prices Dataset from
# https://www.kaggle.com/datasets/yasserh/housing-prices-dataset
# stored in (data/housing.csv).
# The dataset contains the following columns:
# - price (dependent variable or target): Price of the house (in USD).
# - width: Width of the house (in square feet).
# - length: Length of the house (in square feet).
# - bedrooms: Number of bedrooms.
# - bathrooms: Number of bathrooms.
# - stories: Number of stories.
# - mainroad: Whether the house is near a main road (yes or no).
# - guestroom: Whether the house has a guest room (yes or no).
# - basement: Whether the house has a basement (yes or no).
# - hotwaterheating: Whether the house has hot water heating (yes or no).
# - airconditioning: Whether the house has air conditioning (yes or no).
# - parking: Number of parking spots.
# - prefarea: Whether the house is in a preferred area (yes or no).
# - furnishingstatus: Furnishing status of the house (furnished, semi-furnished, unfurnished).
# - quality: An integer value representing the quality of the house.
from typing import Tuple

from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
import pandas as pd
import utils

pd.set_option('display.max_columns', None)  # Display all columns of a DataFrame in pandas
random_state = 42

# LOAD AND VISUALIZE THE DATASET
dataset_file_name = 'data/housing.csv'
integer_independent_vars = ['bedrooms', 'bathrooms', 'stories', 'parking', 'width', 'length', 'quality']
binary_independent_vars = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
nominal_independent_vars = ['furnishingstatus']
independent_vars = integer_independent_vars + binary_independent_vars + nominal_independent_vars
dependent_var = 'price'

# Split the dataset into training and testing sets
(X_train, y_train), (X_test, y_test) = utils.load_dataset_from_csv(dataset_file_name, independent_vars,
                                                                   dependent_var, 0.2, random_state)
# Concatenate X_train and y_train in train_dataset
train_dataset = pd.concat([X_train, y_train], axis=1)
# Visualize the distribution of the dataset
utils.plot_feature_value_distribution(train_dataset)


# DETECT OUTLIERS IN THE DATASET

# Let's detect the outliers in the numerical features using the IQR Tukey’s Fences rule
numerical_features = integer_independent_vars + [dependent_var]
outliers = utils.detect_outliers_iqr(train_dataset[numerical_features], threshold=3.0)
# Shows the values of the outliers detected. Prints the values of the cells that have
# a True value in the 'outliers' DataFrame, indicating that they are outliers.
outliers = train_dataset[numerical_features][outliers].dropna(how='all')  # Drop rows with all NaN values
print("Values of the outliers in the training set:\n", outliers, end='\n\n')

# This is a summary of the outliers.
# - The 'width' and 'length' of one sample are outliers. It does not seem to be erroneous data, so we can scale their
#   values to lessen the impact of their values (StandardScaler or RobustScaler).
# - The 'price' value has two outliers.
#   1) One big value that makes sense (a very expensive house). We can keep it with no scaling
#   because it is the target variable.
#   2) One negative value that is an error. We should remove it from the dataset.

def drop_rows_lower_than(y_set: pd.Series, X_set: pd.DataFrame, threshold: int) ->\
        Tuple[pd.Series, pd.DataFrame]:
    """Drop the rows that have a value lower than the threshold in the target variable."""
    drop_indices_train = y_set[y_set < threshold].index
    return y_set.drop(drop_indices_train), X_set.drop(drop_indices_train)

# Remove the negative values (data cleaning) from y_train and y_test
y_train, X_train = drop_rows_lower_than(y_train, X_train, threshold=0)
y_test, X_test = drop_rows_lower_than(y_test, X_test, threshold=0)


# DETECT MISSING VALUES IN THE DATASET

def show_missing_values(train_set: pd.DataFrame, test_set: pd.DataFrame) -> None:
    """Show the missing values in the training and testing sets."""
    missing_train_values = train_set[train_set.isnull().any(axis=1)]
    print("Missing values in the train dataset:\n", missing_train_values, end='\n\n')
    missing_test_values = test_set[test_set.isnull().any(axis=1)]
    print("Missing values in the test dataset:\n", missing_test_values, end='\n\n')

# Let's detect the missing values in the dataset
# We print the instances that have any missing values
show_missing_values(X_train, X_test)

# There are missing values in the dataset. We will treat them using the same criteria in train and test sets.
# First, if there are 3 or more missing values in a sample, we will remove it.
# Important: We should remove the corresponding rows in the target variable as well.

def drop_rows_multiple_missing_values(X_set: pd.DataFrame, y_set: pd.DataFrame, threshold: int) ->\
        Tuple[pd.DataFrame, pd.DataFrame]:
    """Drop the rows that have a number of missing values greater than or equal to the threshold."""
    drop_indices_train = X_set[X_set.isnull().sum(axis=1) >= threshold].index
    return X_set.drop(drop_indices_train), y_set.drop(drop_indices_train)

X_train, y_train = drop_rows_multiple_missing_values(X_train, y_train, threshold=3)
X_test, y_test = drop_rows_multiple_missing_values(X_test, y_test, threshold=3)

# Impute the missing values the following way:
# - For the integer variables, we will use the median value.
# Let's replace the missing values in the dataset with the median value of the column
integer_imputer = SimpleImputer(strategy='median')
X_train[integer_independent_vars] = integer_imputer.fit_transform(X_train[integer_independent_vars])
X_test[integer_independent_vars] = integer_imputer.transform(X_test[integer_independent_vars])
# - For the binary and nominal variables, we will use the most frequent value.
binary_nominal_imputer = SimpleImputer(strategy='most_frequent')
binary_nominal_features = binary_independent_vars + nominal_independent_vars
X_train[binary_nominal_features] = binary_nominal_imputer.fit_transform(X_train[binary_nominal_features])
X_test[binary_nominal_features] = binary_nominal_imputer.transform(X_test[binary_nominal_features])

# Now, there are no missing values in the dataset
show_missing_values(X_train, X_test)


# SET THE VALID TYPES FOR VARIABLES IN THE DATASET

# Nominal variables should be converted into numbers so that the model can understand them.
# Let's change the "yes" and "no" values to 1 and 0, respectively, for the binary variables
for feature_name in binary_independent_vars:
    X_train[feature_name] = X_train[feature_name].map({'yes': 1, 'no': 0})
    X_test[feature_name] = X_test[feature_name].map({'yes': 1, 'no': 0})
# furnishingstatus cannot be converted into an ordinal integer value, so we will use one-hot encoding
# Let's modify the 'furnishingstatus' feature to be a one-hot encoding
for feature_name in nominal_independent_vars:
    X_train = pd.get_dummies(X_train, columns=[feature_name]).astype(int)
    X_test = pd.get_dummies(X_test, columns=[feature_name]).astype(int)

# Visualize the influence of the features on the target variable
utils.visualize_influence(X_train, y_train)

# Store the curated X_train, X_test, y_train and y_text merged in a single csv file
utils.store_dataset(X_train, y_train, X_test, y_test, 'data/housing_curated.csv')


# MODEL CREATION AND EVALUATION

metrics= utils.create_SDG_regression_model_and_evaluate(X_train, y_train, X_test, y_test, random_state)
utils.show_regression_performance(*metrics)

#########################
#      ACTIVITY
########################

# The previous model does not perform well.
# After reading all the previous code, what is the next step to improve the model?
# Do it and check if the model has increased its performance.

# ANSWER: Scale the data to
# a) improve the performance of the model and
# b) reduce the impact of the outliers in the dataset.

# Let's scale the data using the Robust Scaler, which is robust to outliers.
# Scale all the integer_independent_vars in X_test and X_train
scaler = RobustScaler()
X_train_original, X_test_original = X_train.copy(), X_test.copy()
X_train[integer_independent_vars] = scaler.fit_transform(X_train_original[integer_independent_vars])
X_test[integer_independent_vars] = scaler.transform(X_test_original[integer_independent_vars])

print("Model performance after scaling the data:")
metrics = utils.create_SDG_regression_model_and_evaluate(X_train, y_train, X_test, y_test, random_state)
utils.show_regression_performance(*metrics)

##############################
#      FEATURE ENGINEERING
##############################
# Does it occur to you to perform any feature engineering to improve the model?
# Do one step and check if the model has increased its performance.

# ANSWER: STEP 1. Manually add a new feature 'area' that multiplies the 'width' and 'length' features.
# Important: Scale the data after adding the feature.
X_train['area'] = X_train_original['area'] = X_train_original['width'] * X_train_original['length']
X_test['area'] = X_test_original['area'] = X_test_original['width'] * X_test_original['length']

# Then, scale the data
X_train[['area']] = scaler.fit_transform(X_train[['area']])
X_test[['area']] = scaler.transform(X_test[['area']])

print("Model performance after adding the 'area' feature:")
metrics = utils.create_SDG_regression_model_and_evaluate(X_train, y_train, X_test, y_test, random_state)
utils.show_regression_performance(*metrics)

# The model has subtly improved its performance

# STEP 2. Quality seems influence the price in a quadratic way.
# Let's add a new feature 'quality_squared'.
X_train['quality_squared'] = X_train_original['quality_squared'] = X_train_original['quality'] ** 2
X_test['quality_squared'] = X_test_original['quality_squared'] = X_test_original['quality'] ** 2

# Then, scale the data
X_train[['quality_squared']] = scaler.fit_transform(X_train[['quality_squared']])
X_test[['quality_squared']] = scaler.transform(X_test[['quality_squared']])

print("Model performance after adding the 'quality_squared' feature:")
metrics = utils.create_SDG_regression_model_and_evaluate(X_train, y_train, X_test, y_test, random_state)
utils.show_regression_performance(*metrics)

# The model has significantly improved its performance.

# Let's add new polynomial features to the model to improve its performance.
# Add polynomial features from X_train_original to X_train and X_test
polynomial = PolynomialFeatures(degree=1)
X_train_poly = polynomial.fit_transform(X_train_original)
X_test_poly = polynomial.transform(X_test_original)
# Scale all the polynomial features
X_train = scaler.fit_transform(X_train_poly)
X_test = scaler.transform(X_test_poly)
print("Model performance after adding the polynomial features:")
utils.show_regression_performance(
    *utils.create_SDG_regression_model_and_evaluate(X_train, y_train, X_test, y_test, random_state))

# The model has subtly improved its performance#
# Important: It improves with degree=1 but not with degree=2.
# It is because of over-fitting.

