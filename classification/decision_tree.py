# Desision Tree example.
# We use the Titanic Disaster dataset from
# https://www.kaggle.com/c/titanic/data?select=test.csv
# stored in (data/titanic.csv).
# The dataset has de following features:
# - PassengerId: unique identifier for each passenger.
# - Survived: target variable (0 = No, 1 = Yes).
# - Pclass: ticket class (1 = 1st, 2 = 2nd, 3 = 3rd).
# - Name: name of the passenger.
# - Sex: "male" or "female".
# - Age: age in years.
# - SibSp: number of siblings/spouses aboard.
# - Parch: number of parents/children aboard.
# - Ticket: ticket number.
# - Fare: passenger fare.
# - Cabin: cabin number.
# - Embarked: port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

# QUESTIONS:
# 1) What features are not going to be used and why?
# Answer: PassengerId, Name, Ticket, and Cabin. because they are unique identifiers and may lead to overfitting.
# 2) What features should be transformed and why?
# Answer: Sex and Embarked, because they are nominal features.
# 3) How should they be transformed?
# Answer: - Sex is binary, so 0=female and 1=male is OK.
#         - Embarked is categorical, so we could use one-hot encoding.
#           Do not use LabelEncoder because it would convert it into a numerical feature and the tree could
#           create nodes such as "embarked > 1".
# 4) What features should be scaled and why?
# Answer: None, because CART algorithm does not require scaling.
# 5) Embarked has missing values, what should we do?
# Answer: Nothing, because it will be converted to one-hot encoding and the missing values are correctly
#         represented with false values for all the categories.
# 5) Is there any feature engineering that could be done?
# Answer: Yes, many of them:
# - We could try to create a new feature with the family size.
# - We could try to create to tell siblings and spouses apart.
# - We could try to create to tell parents and children apart.
# We are not going to do it in this example.

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import utils

random_state = 42
pd.set_option('display.max_columns', None)  # Display all columns of a DataFrame in pandas

# LOAD AND CLEAN THE DATASET
dataset_file_name = 'data/titanic.csv'
independent_vars = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']
dependent_var = 'Survived'
class_names = ['Not Survived', 'Survived']

# Load the dataset
original_dataset = pd.read_csv(dataset_file_name)
# Filter out the features that are not going to be used
dataset = original_dataset[independent_vars + [dependent_var]]
print("Filtered dataset:")
print(dataset.head(5), end='\n\n')

# Convert the dataset so that, for the Sex feature, 'male' = 1 and 'female' = 0
pd.options.mode.chained_assignment = None  # to avoid the following false warning
dataset['Sex'] = dataset['Sex'].map({'male': 1, 'female': 0})
# Convert the 'Embarked' feature to one-hot encoding
dataset = pd.get_dummies(dataset, columns=['Embarked'])
print("Transformed dataset:")
print(dataset.head(5), end='\n\n')

# Update the independent variables from the dataset
independent_vars = dataset.columns[dataset.columns != dependent_var]


# DECISION TREE FOR DATA MINING
# We are going to create a decision tree classifier with the whole dataset.
# We do not want to predict the target (machine learning).
# We want to understand the data (data mining).

# Let's create a decision tree with maximum depth 2
model = DecisionTreeClassifier(max_depth=2, random_state=random_state)
model.fit(dataset[independent_vars], dataset[dependent_var])

# Now, we visualize the decision tree
# Take time to analyze the tree and understand the decisions made by the algorithm.
# All the fields in the nodes must be understood.
utils.visualize_decision_tree(model, independent_vars, class_names)

# QUESTIONS:
# 1) Is the dataset balanced?
# Answer: No, because the first node shows that more people did not survive.
# 2) What is the most influential feature on the survival?
# Answer: It is sex, because the first node splits the dataset (more women survived).
# 3) What is the other most influential features?
# Answer: Pclass and fare, because they are used in the two next modules.
# 4) There are decision rules that can be extracted from the tree. For example:
#    "if <condition> then survived = 1". Write all of them.
# Answer: if sex == female and pclass <= 2 then survived = 1
#         if sex == female and pclass == 3 then survived = 0
#         if sex == male then survived = 0


# DECISION TREE FOR MACHINE LEARNING

# Split the dataset into training and testing sets
(X_train, y_train), (X_test, y_test) = utils.split_dataset(dataset, independent_vars,
                                                           dependent_var, 0.2, random_state)
# Create and train the model
model = DecisionTreeClassifier(random_state=random_state, max_depth=2)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy_value = accuracy_score(y_test, y_pred)
f1_score_value = f1_score(y_test, y_pred)
print(f"Accuracy: {accuracy_value:.2f}.\nF1 Score: {f1_score_value:.2f}.")

# QUESTIONS:
# 1) What score is higher? Why?
# Answer: The accuracy is higher because the dataset is imbalanced.
# 2) Does it make sense to compute AUC-ROC using the model?
# Answer: No, because decision trees do not output probabilities.
#         Remember, that AUC is based on changing the threshold of the probabilities and compute
#         the area of the ROC curve created between the TPR and FPR.


# DIFFERENT MAX DEPTHS

# Create and train the model
print("Creating decision trees with different max depths:")
for max_depth in [2, 6, 11]:
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy_value = accuracy_score(y_test, y_pred)
    f1_score_value = f1_score(y_test, y_pred)
    print(f"\tMax Depth: {max_depth}. Accuracy: {accuracy_value:.2f}. F1 Score: {f1_score_value:.2f}.")

# QUESTIONS:
# 1) What happens when the max depth is 2? Why?
# Answer: The model is underfitting because it is too simple.
# 2) What happens when the max depth is 6? Why?
# Answer: The model is better because it is more complex.
# 3) What happens when the max depth is 11? Why?
# Answer: The model is overfitting because it is too complex.
# 4) How would you choose the best max depth?
# Answer: Using validation set. You should do this with any hyperparameter.

