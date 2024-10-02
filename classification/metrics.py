# This example shows different metrics for classifiers.
# We use the Breast Cancer Wisconsin (Diagnostic) Data Set

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import utils
import models

random_state = 42

# LOAD AND VISUALIZE THE DATASET
dataset_file_name = 'data/cancer.csv'
independent_vars = ['radius_mean', 'texture_mean']
dependent_var = 'diagnosis'

dataset = pd.read_csv(dataset_file_name)
# Replace diagnosis values with 0 and 1
dataset[dependent_var] = dataset[dependent_var].map({'M': 1, 'B': 0})
# Describe the dependent variable of the dataset
print(f"Percentage of positive labels: {dataset[dependent_var].mean() * 100:.2f}%.")

# Split the dataset into training and testing sets
(X_train, y_train), (X_test, y_test) = utils.split_dataset(dataset, independent_vars,
                                                           dependent_var, 0.2, random_state)

# Scale the dataset with a Standard Scaler
X_train, X_test = utils.scale_X_dataset(X_train, X_test, StandardScaler)

# DIFFERENT BINARY CLASSIFICATION MODELS

lr_model = LogisticRegression(random_state=random_state)
lr_model.fit(X_train, y_train)
always_true_model = models.AlwaysTrue()
always_false_model = models.AlwaysFalse()
random_model = models.RandomModel(random_state)
lr_0_5_model = models.ThresholdModel(lr_model, 0.5)
lr_0_8_model = models.ThresholdModel(lr_model, 0.8)
lr_0_2_model = models.ThresholdModel(lr_model, 0.2)

models = [always_true_model, always_false_model, random_model, lr_0_5_model, lr_0_8_model, lr_0_2_model]

# QUESTIONS about accuracies:
# 1) What is the best accuracy from always_true_model and always_false_model?
# Answer: always_false_model, because only 37% of the labels are positive.
# 2) Is the accuracy of random_model better than 0.5?
# Answer: We cannot know, but it would probably be because the dataset is imbalanced.
# 3) What is the best threshold between 0.8 and 0.2 for the logistic regression model?
# Answer: The best threshold is 0.8, because there are more 0s than 1s in the dataset
# (with 0.8 we make it more difficult to predict the positive class).

# Uncomment the following code and see if your answers are correct
# accuracies = utils.compute_metrics(models, X_test, y_test, accuracy_score)
# utils.show_metrics('Accuracy', accuracies, models)


# QUESTIONS about precision and recall:
# 1) What are going to be the precision and recall of always_false_model?
# Answer: precision = 0 and recall = 0, because there are no positives and both metrics measure positives.
# 2) What are going to be the precision and recall of always_true_model?
# Answer: precision = 0.37 because is the percentage of positives.
#         recall = 1, all the existing positives are correctly classified.
# 3) Are precision and recall of the random model going to be greater than 0.5?
# Answer: precision probably lower than 0.5 because only 37% are positives.
#         recall around 0.5 because, for each positive, we have 0.5 change to guess it.
# 4) What is the threshold with the best precision?
# Answer: 0.8 because it reduces the number of positives (only takes the most certain ones).
# 5) What is the threshold with the best recall?
# Answer: 0.2 because it maximizes the number of samples classified as positive.

# Uncomment the following code and see if your answers are correct
#precisions = utils.compute_metrics(models, X_test, y_test, precision_score)
#recalls = utils.compute_metrics(models, X_test, y_test, recall_score)
#utils.show_metrics('Precision', precisions, models)
#utils.show_metrics('Recall', recalls, models)

# QUESTIONS about F1 score:
# 1) What is the F1 score of always_false_model?
# Answer: 0, because precision and recall are 0 (only considers positives).
# 2) What is the F1 score of always_true_model?
# Answer: not very high, because recall is 1 but precision is 0.37.
# 3) What is the F1 score of random_model?
# Answer: not very high, because precision is low and recall is around 0.5.
# 4) What is the threshold with the best F1 score?
# Answer: 0.5, because it is the most balanced threshold.

# Uncomment the following code and see if your answers are correct
#f1_scores = utils.compute_metrics(models, X_test, y_test, f1_score)
#utils.show_metrics('F1-score', f1_scores, models)


# QUESTIONS about the AUC:
# 1) What is the AUC of always_true_model?
# Answer: 0.5, because it measures a trade-off of positive and negative (mis)classifications
# 2) What is the AUC of always_false_model?
# Answer: 0.5, for the same reason
# 3) What is the AUC of random_model?
# Answer: around 0.5, because it is random (and it measures positives and negatives so it does not
#         depend on dataset balance).
# 4) Does it make sense to compute the AUC of the logistic regression model for different thresholds?
# Answer: No, because AUC measures the performance for *changing* thresholds
# 5) What will the AUC of the logistic regression model be?
# Answer: close to 1 because the model is quite good.

# Uncomment the following code and see if your answers are correct
#auc_models = [always_true_model, always_false_model, random_model, lr_0_5_model]
#auc_scores = utils.compute_AUCs(auc_models, X_test, y_test)
#utils.show_metrics('AUC score', auc_scores, auc_models)
