# We tackle two things: ensemble methods and how to compare the performance of different models.
# We use the Titanic Disaster dataset from
# https://www.kaggle.com/c/titanic/data?select=test.csv
# stored in (data/titanic.csv).


import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import resample
from time import time

import utils

random_state = 42

# LOAD AND CLEAN THE DATASET
dataset_file_name = 'data/titanic.csv'
independent_vars = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']
dependent_var = 'Survived'
class_names = ['Not Survived', 'Survived']

dataset, independent_vars = utils.load_clean_titanic_dataset(dataset_file_name, independent_vars, dependent_var)

# Split the dataset into training and testing sets
(X_train, y_train), (X_test, y_test) = utils.split_dataset(dataset, independent_vars, dependent_var,
                            # we choose a train size = 0.4 test size of 0.6 on purpose to have higher variance
                                                    0.6, random_state)

# DIFFERENT BINARY CLASSIFICATION MODELS WITH THE DEFAULT HYPERPARAMETERS

# Please, note that we are not tuning the hyperparameters of the models.
# This is important, because the performance of the models could be improved by tuning the hyperparameters.
# Particularly, the Random Forest and XGBoost models have many hyperparameters that could be tuned and improved.

print(f"{'-'*5} Single-value evaluation method {'-'*5}")
time_before = time()
# Let's different DT-based models
dt_model = DecisionTreeClassifier(random_state=random_state)
dt_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(random_state=random_state)
rf_model.fit(X_train, y_train)

xgb_model = xgb.XGBClassifier(random_state=random_state)
xgb_model.fit(X_train, y_train)

# Evaluate the models
utils.evaluate_models([dt_model, rf_model, xgb_model], X_test, y_test)
print(f"Time elapsed: {time() - time_before:.2f} seconds.")


# QUESTIONS:
# 1) Can we state that XGBoost is the best model?
# Answer: It seems so, because it has the highest accuracy and F1 score.
# ACTIVITY
# Set the random_state to None and run the code many times. Is XGBoost always the best model?
# 2) What is happening?
# Answer: The performance of the models is changing because the follow a stochastic process.
# 3) What could we do to improve the comparison?
# Answer: We could compute confidence intervals of the metrics and see if there is a significant difference between them.


# METHOD ONE: TRAIN AND EVALUATE THE MODELS N TIMES AND COMPUTE THE 95% CONFIDENCE INTERVALS

# Evaluate the performance of a desired model N times using cross validation
print(f"\n{'-'*5} Re-train and re-evaluate method {'-'*5}")
time_before = time()
n_times = 30
accuracies = dict()
f1_scores = dict()
for _ in range(n_times):
    (temp_X_train, temp_y_train), (temp_X_test, temp_y_test) = utils.split_dataset(dataset, independent_vars, dependent_var,
                                                               0.6, random_state=None)
    dt_model = DecisionTreeClassifier(random_state=None)
    dt_model.fit(temp_X_train, temp_y_train)
    rf_model = RandomForestClassifier(random_state=None)
    rf_model.fit(temp_X_train, temp_y_train)
    xgb_model = xgb.XGBClassifier(random_state=None)
    xgb_model.fit(temp_X_train, temp_y_train)
    models = [dt_model, rf_model, xgb_model]
    metrics = utils.evaluate_models(models, temp_X_test, temp_y_test, verbose=False)
    for model_name, (accuracy, f1_score_value) in zip(models, metrics):
        accuracies.setdefault(model_name.__class__.__name__, []).append(accuracy)
        f1_scores.setdefault(model_name.__class__.__name__, []).append(f1_score_value)

# Compute and visualize the confidence intervals of each model and metric
for model_name in accuracies:
    accuracy_mean, accuracy_confidence_interval = utils.confidence_interval(accuracies[model_name], 0.95)
    f1_score_mean, f1_score_confidence_interval = utils.confidence_interval(f1_scores[model_name], 0.95)
    print(f"Model: {model_name}.")
    print(f"\tAccuracy mean: {accuracy_mean:.4f}. CI: {accuracy_confidence_interval}.")
    print(f"\tF1 Score: {f1_score_mean:.4f}. CI: {f1_score_confidence_interval}.")

print(f"Time elapsed: {time() - time_before:.2f} seconds.")

# QUESTIONS:
# 1) Are there differences with the previous values?
# Answer: Yes, they are not the same values.
# 2) Why?
# Answer: Because, in this method we create different models (not just one) and average the results.
# 3) What method do you think is better?
# Answer: The second one, because it is more reliable and less random.
# 4) Is there any modification of the best, average, and worst models?
# Answer: There might be. It depends on the randomness of the process.
#         In my execution, the average values make XGBoost the best model, then RF and finally DT.
# 5) Are there significant differences between the models?
# Answer: There might be. It depends on the randomness of the process.
#         In my execution, the confidence intervals of the accuracy and F1 score of the models overlap
#         for both metrics and all the models.
#         Thus, there are NOT significant differences between the models.
#         This is because they are very random, since we took 40% for training and 60% for testing.


# METHOD TWO: USE BOOTSTRAP ON THE TEST SET

# Notice. This method is used when retraining the models is so expensive that we cannot afford to retrain them N times.
# A common example is a deep learning model that takes hours to train.
# In this case, we use the test set to compute the confidence intervals of the metrics
# using a method called bootstrapping.

print(f"\n{'-'*5} Bootstrapping method {'-'*5}")
time_before = time()

dt_model = DecisionTreeClassifier(random_state=random_state)
dt_model.fit(X_train, y_train)
rf_model = RandomForestClassifier(random_state=random_state)
rf_model.fit(X_train, y_train)
xgb_model = xgb.XGBClassifier(random_state=random_state)
xgb_model.fit(X_train, y_train)
models = [dt_model, rf_model, xgb_model]

# Perform bootstrapping
n_bootstrap_samples = 1000
bootstrap_accuracies = dict()
bootstrap_f1_scores = dict()
for _ in range(n_bootstrap_samples):
    # Resample with replacement, both X_test and the corresponding y_test at the same time (important)
    resampled_X_test, resampled_y_test = resample(X_test, y_test, replace=True)
    # Calculate the metrics for each model and store it in the two dictionaries
    for model in models:
        resampled_y_pred = model.predict(resampled_X_test)
        accuracy = accuracy_score(resampled_y_test, resampled_y_pred)
        f1_score_value = f1_score(resampled_y_test, resampled_y_pred)
        bootstrap_accuracies.setdefault(model.__class__.__name__, []).append(accuracy)
        bootstrap_f1_scores.setdefault(model.__class__.__name__, []).append(f1_score_value)

# Compute and visualize the confidence intervals of each model and metric
for model_name in bootstrap_accuracies:
    accuracy_mean, accuracy_confidence_interval = utils.confidence_interval(bootstrap_accuracies[model_name], 0.95)
    f1_score_mean, f1_score_confidence_interval = utils.confidence_interval(bootstrap_f1_scores[model_name], 0.95)
    print(f"Model: {model_name}.")
    print(f"\tAccuracy mean: {accuracy_mean:.4f}. CI: {accuracy_confidence_interval}.")
    print(f"\tF1 Score: {f1_score_mean:.4f}. CI: {f1_score_confidence_interval}.")

print(f"Time elapsed: {time() - time_before:.2f} seconds.")

# QUESTIONS:
# 1) Are there differences with the previous values? Why?
# Answer: Yes, they are not the same because they use different method of estimation
#         and, very important, we do not have many data.
#         With sufficient data, the two methods should give closer results.
# 2) What method takes longer? Why?
# Answer: In this execution the last one because it predicts and evaluates on 1000 different bootstrapped samples
#         of the test set?
# 3) In deep learning scenarios, do you think there will be similar execution-time differences?
# Answer: Not in deep models because training is *way much more* expensive than predicting.
