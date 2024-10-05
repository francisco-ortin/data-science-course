# We will perform customer segmentation using the KMeans algorithm.
# The dataset contains information about customers such as their annual income, spending
# score, age and gender.
# We will use the KMeans algorithm to cluster the customers into different groups.
# The goal is to understand the different customer segments and their characteristics.
# The dataset contains the following columns: 'Annual Income (k$)', 'Spending Score (1-100)', 'Age' and 'Gender'.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import seaborn as sns
from warnings import simplefilter
simplefilter(action='ignore')  # ignore all the warnings


random_state = 42

dataset = pd.read_csv('data/customers.csv')
feature_names = ['Annual Income (k$)', 'Spending Score (1-100)', 'Age', 'Gender']


# Replace 'Gender' feature: Male with 1 and Female with 0
dataset['Gender'] = dataset['Gender'].map({'Male': 1, 'Female': 0})

# Show the dataset information
print(dataset.head())
print(dataset.info())
print(dataset.describe())


def show_feature_distribution(X: pd.DataFrame, feature_name: str, title: str, bins: int) -> None:
    """Shows the distribution of a feature in the dataset"""
    plt.figure(figsize=(10, 6))
    plt.hist(X[feature_name], bins=bins, color='skyblue')
    plt.title(title)
    plt.xlabel(feature_name)
    plt.ylabel('Count')
    plt.show()


show_feature_distribution(dataset, 'Annual Income (k$)', 'Distribution of Annual Income', bins=20)
show_feature_distribution(dataset, 'Spending Score (1-100)', 'Distribution of Spending Score', bins=20)
show_feature_distribution(dataset, 'Age', 'Distribution of Age', bins=10)
show_feature_distribution(dataset, 'Gender', 'Distribution of Gender', bins=2)


# Let's compute the silhouette scores for k=2 to k=10
def compute_silhouette_scores(X: np.array, from_k: int, to_k: int) -> np.array:
    scores = []
    for k in range(from_k, to_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        clusters = kmeans.fit_predict(X)
        score = silhouette_score(X, clusters)
        scores.append(score)
    # Plot the silhouette scores
    plt.figure(figsize=(8, 4))
    plt.plot(range(from_k, to_k + 1), scores, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal k')
    plt.show()
    return scores


# Compute the optimal number of clusters using the silhouette score
silhouette_scores = compute_silhouette_scores(dataset, 2, 10)
score_per_k = list(enumerate(silhouette_scores, 2))
print("Silhouette Scores (k, score):", score_per_k)
# k is set to the maximum value
n_clusters = max(score_per_k, key=lambda x: x[1])[0]
print("Optimal number of clusters:", n_clusters)

# Let's fit the KMeans model with the optimal number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
clusters = kmeans.fit_predict(dataset)

# Apply t-SNE to reduce the dataset to 2D
tsne = TSNE(n_components=2, random_state=random_state)
X_tsne = tsne.fit_transform(dataset)


# Let's visualize the clusters of customers to see if the are well separated
def visualize_2D(X: np.array, y: np.array, title: str) -> None:
    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("hsv", 10)
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette=palette, legend='full', alpha=0.5)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

visualize_2D(X_tsne, clusters, 'Clusters of Customers')


# Show the distribution of all the features for each cluster
def show_cluster_distribution(X: pd.DataFrame, clusters: np.array, n_clusters: int, feature_name: str, bins: int) -> None:
    plt.figure(figsize=(10, 6))
    for cluster in range(n_clusters):
        sns.distplot(X[clusters == cluster][feature_name], bins=bins, label=f'Cluster {cluster}', hist=False)
    plt.title(f'Distribution of {feature_name} for each Cluster')
    plt.xlabel(feature_name)
    plt.ylabel('Density')
    plt.legend()
    plt.show()


show_cluster_distribution(dataset, clusters, n_clusters, 'Annual Income (k$)', bins=20)
show_cluster_distribution(dataset, clusters, n_clusters, 'Spending Score (1-100)', bins=20)
show_cluster_distribution(dataset, clusters, n_clusters, 'Age', bins=10)
show_cluster_distribution(dataset, clusters, n_clusters, 'Gender', bins=2)

# Show information about the clusters
for cluster in range(n_clusters):
    print(f'Cluster {cluster}:')
    print(dataset[clusters == cluster].describe())



# ACTIVITY: This file shows you impotant information about the customers dataset.
# You must perform customer segmentation.
# Document the profile of customer for each cluster.
# Having a better understanding of the customers segments, a company could make better and more informed decisions.
# * Cluster 0. Write the customer profile for this cluster.
# * Cluster 1. Write the customer profile for this cluster.
# * Cluster 2. Write the customer profile for this cluster.
# * Cluster 3. Write the customer profile for this cluster.
# * Cluster 4. Write the customer profile for this cluster.
# * Cluster 5. Write the customer profile for this cluster.

# ANSWER:
# * Cluster 0.
#   Highest age: 50-70.
#   Average spending score and annual income.
# * Cluster 1. Write the customer profile for this cluster.
#   Mostly men.
#   Middle age.
#   Low spending score.
#   High annual income.
# * Cluster 2. Write the customer profile for this cluster.
#   Young customers.
#   High spending score.
#   Low annual income.
# * Cluster 3. Write the customer profile for this cluster.
#   Mostly women.
#   Young customers.
#   Average spending score and annual income.
# * Cluster 4. Write the customer profile for this cluster.
#   Middle-low age.
#   High spending score.
#   High annual income.
# * Cluster 5. Write the customer profile for this cluster.
#   Middle age.
#   Low spending score and annual income.
