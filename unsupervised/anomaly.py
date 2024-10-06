# The KDD Cup 1999 dataset (KDDCUP99) is one of the most well-known datasets used for evaluating
# machine learning models for network intrusion detection.
# It contains a variety of network traffic data, where each record is labeled as either normal or as an attack type.
# It has 41 features, including the duration of the connection, the protocol type, the service, the flag, and more.
# The labels are divided into 5 categories: normal, probe, dos, u2r, and r2l.
# Most of the records are normal, and the rest are attacks.

# Let's see how an anomaly detection analysis can help us detect outliers in this dataset.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


random_state = 42
pd.set_option('display.max_columns', None)  # Display all columns of a DataFrame in pandas
NUMBER_OF_SAMPLES = 1_000
categorical_features = ['protocol_type', 'service', 'flag']


def load_kddcup99_data(number_of_samples: int | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Loads the CSV file, clens the dataset and returns a selection of the X and y data."""
    df = pd.read_csv('data/kddcup99.csv')
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    # Create y as 0 if "labels" is normal, 1 if attack
    y = df['labels'].apply(lambda x: 0 if x == "normal" else 1)
    # drop the "labels" column
    df = df.drop('labels', axis=1)
    # Select the first number_of_samples samples
    if number_of_samples is not None:
        df = df.iloc[:number_of_samples]
        y = y.iloc[:number_of_samples]
    return df, y


# Loads the dataset
X_original, y = load_kddcup99_data(NUMBER_OF_SAMPLES)
print(X_original.describe(include='all'))

# Treat missing values
missing_values = X_original.isnull().sum().sum()
print(f"Missing values in the dataset: {missing_values}.")
assert missing_values == 0, "There are missing values in the dataset. You must tackle them."

# Treat categorical features
X = pd.get_dummies(X_original, columns=categorical_features, drop_first=True)


# Let's see the distribution of the numerical features to see if Tukey's fences could be applied
def show_numerical_feature_distribution(dataframe: pd.DataFrame):
    """Show in different figures the distribution of the numerical features.
    Create one figure for a group of 8 numerical features."""
    numerical_features = dataframe.select_dtypes(include='number').columns
    for i in range(0, len(numerical_features), 8):
        plt.figure(figsize=(16, 8))
        for j, feature in enumerate(numerical_features[i:i + 8]):
            plt.subplot(2, 4, j + 1)
            sns.histplot(dataframe[feature], kde=True)
            plt.title(feature)
        plt.tight_layout()
        plt.show()



show_numerical_feature_distribution(X)


# # QUESTIONS:
# # 1) To which features can we apply Tukey's fences to detect outliers?
# # Answer: None of them.
# # 2) Why?
# # Answer: Tukey's fences are used to detect outliers in numerical features that are symmetric and unimodal.
# #         Only 'is_host_login' and 'num_outbound_cmds' meet these criteria,
# #         but they have a single value for all samples (no outlier exists).
#
#
# def apply_isolation_forest(df):
#     iso_forest = IsolationForest(contamination=0.03, random_state=42)
#     outliers = iso_forest.fit_predict(df)
#     # Modify inliers 1 to 0 and outliers -1 to 1
#     outliers[outliers == 1] = 0
#     outliers[outliers == -1] = 1
#     return outliers
#
#
# y_if_outliers = apply_isolation_forest(X)
# print(f"Isolation Forest outliers: {sum(y_if_outliers)}/{len(y_if_outliers)} samples "
#       f"({sum(y_if_outliers)/len(y_if_outliers):.2f}%).")
# print(f"Actual number of outliers: {sum(y)}/{len(y)} samples ({sum(y)/len(y):.2f}%).")
#
#
# def plot_tsne(X_dataframe: pd.DataFrame, y_iso_outliers: np.array, y_actual_outliers: np.array) -> None:
#     """
#     Plot two figures: actual outliers (left) and Isolation Forest outliers (right).
#     Plot the t-SNE of the dataset (all the points) on green in both plots.
#     Left plot: those points with the y value in 1 (actual outliers) are plotted in red.
#     Right plot: Those points in iso_outliers are plotted in blue (Isolation Forest) outliers.
#     :param X_dataframe: All the dataset
#     :param y_iso_outliers: Isolation Forest outliers
#     :param y_actual_outliers: Actual outliers
#     """
#     tsne = TSNE(n_components=2, random_state=random_state)
#     X_tsne = tsne.fit_transform(X_dataframe)
#     plt.figure(figsize=(16, 8))
#     plt.subplot(1, 2, 1)
#     plt.scatter(X_tsne[y_actual_outliers == 0, 0], X_tsne[y_actual_outliers == 0, 1], c='green', label='Normal')
#     plt.scatter(X_tsne[y_actual_outliers == 1, 0], X_tsne[y_actual_outliers == 1, 1], c='red', label='Actual Outliers')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.title('Actual Outliers')
#     plt.legend()
#     plt.subplot(1, 2, 2)
#     plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c='green', label='Normal')
#     plt.scatter(X_tsne[y_iso_outliers == 1, 0], X_tsne[y_iso_outliers == 1, 1], c='blue',
#                 label='Isolation Forest Outliers')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.title('Isolation Forest Outliers')
#     plt.legend()
#     plt.show()
#
#
# plot_tsne(X, y_if_outliers, y.values)
#
# # QUESTIONS:
# # 3) Is the Isolation Forest algorithm detecting the outliers well?
# # Answer: Some of them are detected, but not all.
# # 4) Modify the contamination parameter in the Isolation Forest algorithm to see if you can improve the results.
# # Answer: The contamination parameter is set to 0.03, which is the percentage of outliers in the dataset.
# #         Bigger values will detect more outliers, but also more false positives.
# # 5) How can you evaluate the effectiveness of the Isolation Forest algorithm?
# # Answer: By comparing the actual outliers with the detected outliers.
# #         A good metric is the F1 score, which combines precision and recall.
# #         Accuracy is not good for imbalanced datasets like this one.
# # 6) (activity) Compute that metric and show the result in our example.
#
#
# # SOLUTION:
# from sklearn.metrics import f1_score
# f1_score_iso = f1_score(y, y_if_outliers)
# print(f"F1 score for Isolation Forest: {f1_score_iso:.4f}.")
