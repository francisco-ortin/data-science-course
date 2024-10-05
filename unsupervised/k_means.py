# The iris dataset is a popular dataset in machine learning.
# It contains 150 samples of iris flowers.
# Each sample has 4 features: sepal length, sepal width, petal length, and petal width.
# The dataset is divided into 3 classes: setosa, virginica, and versicolor.

# In this exercise, you will use the K-Means algorithm to cluster the iris dataset.
# You will also use the elbow method and silhouette score to find the optimal number of clusters.

from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)  # ignore all future warnings

random_state = 42

# Load the iris dataset
def load_iris_data() -> Tuple[np.array, np.array]:
    """
    Load and shuffle the iris dataset
    :return: The dataset features (X) and labels (y)
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    np.random.seed(random_state)
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    return X, y

X_dataset, y_dataset = load_iris_data()
# Show the first 5 rows of the dataset and the corresponding label
for i in range(5):
    print(f"Sample: {X_dataset[i]} \t Label: {y_dataset[i]}.")


# Plot the elbow method to find the optimal number of clusters
def plot_elbow(X: np.array, from_k: int, to_k: int) -> None:
    """
    Plot the elbow method to find the optimal number of clusters
    :param X: the dataset
    :param from_k: the minimum number of clusters
    :param to_k: the maximum number of clusters (included)
    """
    distortions = []
    for k in range(from_k, to_k + 1):
        kmeans = KMeans(n_clusters=k, init='random', n_init='auto', random_state=random_state)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, to_k + 1), distortions, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Distortion')
    plt.title('Elbow Method for Optimal k')
    plt.show()

plot_elbow(X_dataset, from_k=1, to_k=10)

# QUESTIONS:
# 1) What is the k value for the elbow point in the plot?
# Answer: The elbow point is at k=2.
# 2) Does it coincide with the number of classes in the iris dataset?
# Answer: No, the iris dataset has 3 classes, but the elbow point is at k=2.
# 3) What is the cause?
# Answer: There are 3 types of iris flowers in the dataset,
#         but their sepal and petal structure better define two clusters.
#         That is, there might be other features that tell the difference between the 3 classes.


# Let's compute the clusters using k=3
def compute_clusters(X: np.array, n_clusters: int) -> np.array:
    """
    Compute the clusters using K-Means
    :param X: the dataset
    :param n_clusters: the number of clusters
    :return: the cluster labels for each sample in the dataset
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    return kmeans.fit_predict(X)


clusters = compute_clusters(X_dataset, n_clusters=3)


def plot_clusters(X: np.array, clusters: np.array, y: np.array, title1: str, title2: str) -> None:
    """
    Plot the clusters (left) and the original dataset (right) in the same figure
    :param X: the dataset
    :param clusters: the cluster labels
    :param y: the original labels (setosa, virginica, versicolor)
    :param title1: title for the clusters plot
    :param title2: title for the original dataset plot
    """
    # reduce the features to 2D using PCA
    pca = PCA(2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(12, 6))
    # first plot
    plt.subplot(1, 2, 1)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50)  # s=50 is the size of the points
    plt.title(title1)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    # second plot
    plt.subplot(1, 2, 2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=50)
    plt.title(title2)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    # show the figure
    plt.show()


plot_clusters(X_dataset, clusters, y_dataset, 'K-Means Clustering', 'Original Dataset')

# QUESTIONS:
# 4) Is the k-means working well?
# Answer: The k-means algorithm is working well because the closest flowers are
#         clustered in the same group (the cluster they structurally belong to).
# 5) How well do the clusters separate the iris dataset?
# Answer: One cluster is well separated, but there are minor errors
# in the other two.


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


silhouette_scores = compute_silhouette_scores(X_dataset, 2, 10)
print("Silhouette Scores (k, score):", list(enumerate(silhouette_scores, 2)))

# QUESTIONS:
# 6) What is the best k value according to the silhouette score?
# Answer: The best k value according to the silhouette score is k=2.
# 7) Does it coincide with the elbow method?
# Answer: Yes, the silhouette score and the elbow method both suggest k=2.
# 8) Does it coincide with the number of classes in the iris dataset? Why?
# Answer: No, the iris dataset has 3 classes, but the silhouette score suggests k=2.
#         As mentioned, there are 3 types of iris flowers in the dataset,
#         but their sepal and petal structure better define two clusters.
#         That is, there might be other features that tell the difference between the 3 classes.

