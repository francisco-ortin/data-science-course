# This lab is about dimensionality reduction.
# We will use the MNIST dataset, which is a dataset of 28x28 grayscale images of handwritten digits (0 to 9).
# The dataset contains 60,000 training images and 10,000 testing images.
# Each image is a 28x28 pixel square (784 features/pixels in total).
# Each pixel has a value between 0 and 255, representing the grayscale intensity.

# Read the notebook and answer the questions in the comments.

from typing import Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score

TRAIN_FILE_NAME = 'data/mnist_train.csv.zip'
TEST_FILE_NAME = 'data/mnist_test.csv.zip'
ramdom_state = 42

# Load MNIST dataset from mnist_test.csv
def load_data(csv_zip_file_name: str) -> Tuple[np.array, np.array]:
    # Load the dataset from a zip CSV file
    data = pd.read_csv(csv_zip_file_name, compression='zip')
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    return X, y



X_train, y_train = load_data(TRAIN_FILE_NAME)
X_test, y_test = load_data(TEST_FILE_NAME)

# visualize some examples of the dataset
def visualize_samples(X: np.array, y: np.array, title: str, n_samples: int) -> None:
    fig, axes = plt.subplots(1, n_samples, figsize=(10, 3))
    for i in range(n_samples):
        axes[i].imshow(X[i].reshape(28, 28), cmap='gray')
        axes[i].axis('off')
        # show the digit label
        axes[i].set_title(f'Digit: {y[i]}')
    plt.suptitle(title)
    plt.show()


visualize_samples(X_train, y_train, 'Sample digits from MNIST dataset', 10)

# perform PCA reduction to 2D
def pca_reduction(X: np.array, n_components: Optional[int]) -> Tuple[np.array, np.array, PCA]:
    """
    Perform PCA reduction to n_components
    :param X: The source dataset to be reduced
    :param n_components: the number of components (dimensions) to reduce to
    :return: The reduced dataset, the explained variance ratio per component and the PCA model
    """
    pca = PCA(n_components=n_components, random_state=ramdom_state)
    X_pca = pca.fit_transform(X)
    return X_pca, pca.explained_variance_ratio_, pca


X_train_PCA, _, _ = pca_reduction(X_train, 2)

# Visualize the reduced dataset in two dimensions using PCA
def visualize_2D(X: np.array, y: np.array, title: str) -> None:
    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("hsv", 10)
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette=palette, legend='full', alpha=0.5)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


SAMPLES_TO_PLOT = 5_000
visualize_2D(X_train_PCA[:SAMPLES_TO_PLOT], y_train[:SAMPLES_TO_PLOT], 'PCA reduction of MNIST to 2D')

# QUESTIONS:
# 1) Identify the two main digits that, after PCA reduction, are the easiest to separate
# Answer: 0 and 1.
# 2) Why do you think these two digits are the easiest to separate?
# Answer: The digits 0 and 1 are the easiest to separate because they have a distinct shape compared to other digits.


# t-SNE is another dimensionality reduction technique that is particularly well-suited for visualization purposes.
# t-SNE stands for t-distributed stochastic neighbor embedding.
# t-SNE is a nonlinear dimensionality reduction technique that preserves local structure in the data.

# Let's visualize the reduced dataset in two dimensions using t-SNE
def tsne_reduction(X: np.array, n_components: int) -> Tuple[np.array, TSNE]:
    tsne = TSNE(n_components=n_components, random_state=ramdom_state)
    X_tsne = tsne.fit_transform(X)
    return X_tsne, tsne

X_train_t_SNE, _ = tsne_reduction(X_train[:SAMPLES_TO_PLOT], 2)
visualize_2D(X_train_t_SNE, y_train[:SAMPLES_TO_PLOT], 't-SNE reduction of MNIST to 2D')

# QUESTIONS:
# 3) Compare the PCA and t-SNE visualizations. For this dataset, which technique seems to provide better separation?
# Answer: t-SNE provides better separation than PCA for this dataset.
# The t-SNE visualization shows more distinct clusters for each digit.
# 4) Why?
# Answer: Because digit classification seems to be non-linear (t-SNE is a nonlinear dimensionality reduction technique)
# 5) What are the hardest digits to separate by t-SNE?
# Answer: The hardest digits to separate by t-SNE are 9 and 4.
# 6) Why?
# Answer: These digits have a similar shape and are more likely to be confused by the t-SNE algorithm.


# Show the explained variance ration for PCA reduction n dimensions
# and select the smaller dimension with 95% explained variance
def get_pca_n_components_with_explained_variance(X: np.array, min_explained_variance: float) -> int:
    """
    Get the number of PCA components that explain at least min_explained_variance of the variance
    :param X: dataset
    :param min_explained_variance: minimum explained variance ratio
    :return: the number of components that explain at least min_explained_variance of the variance
    """
    X_train_PCA, explained_variances, _ = pca_reduction(X, None)
    print('Explained variance ratios per dimensions (components):')
    min_components = 0
    for n in range(len(explained_variances)):
        explained_variance = explained_variances[:n+1].sum()
        print(f'\tExplained variance ratio for {n+1} dimensions: {explained_variance:.4f}.')
        min_components = n
        if explained_variance >= min_explained_variance:
            break
    return min_components

min_explained_variance = 0.95
min_components = get_pca_n_components_with_explained_variance(X_train, min_explained_variance)
print(f'The dataset can be reduced from {X_train.shape[1]} to {min_components} dimensions with '
      f'{min_explained_variance*100:.2f}% explained variance.')
X_train_PCA, _, pca_model = pca_reduction(X_train, min_components)


# Perform a PCA reconstruction of the first 10 instances of the reduced dataset
# Visualize the differences between the original and the reconstructed images
X_reconstructed = pca_model.inverse_transform(X_train_PCA)
visualize_samples(X_train, y_train, 'Sample digits from MNIST dataset', 10)
visualize_samples(X_reconstructed, y_train, 'Reconstructed sample digits from MNIST dataset', 10)

# QUESTIONS:
# 7) Do you see any difference between the original and the reconstructed images?
# Answer: Yes, the reconstructed images are blurrier than the original ones.
# 8) Why?
# Answer: The PCA reconstruction is an approximation of the original data using fewer dimensions,
#         since some information is lost (it works like a compression algorithm).


# Create one classifier for the reduced dataset (previous point) and the original one
# Evaluate the performance (accuracy) of both models
def evaluate_classifier_performance(X_train_p: np.array, y_train_p: np.array, X_test_p: np.array, y_test_p: np.array,
                                    classifier) -> float:
    """
    Build and evaluate the performance of a classifier on a given dataset
    :return: accuracy of the classifier
    """
    classifier.fit(X_train_p, y_train_p)
    y_pred = classifier.predict(X_test_p)
    accuracy = accuracy_score(y_test_p, y_pred)
    return accuracy


print("Building and evaluating RF classifier models...")
SAMPLES_TO_TRAIN_THE_CLASSIFIER = 1_000  # to speed up the process
rf_model = RandomForestClassifier(random_state=ramdom_state)
accuracy_original = evaluate_classifier_performance(X_train[:SAMPLES_TO_TRAIN_THE_CLASSIFIER],
                                            y_train[:SAMPLES_TO_TRAIN_THE_CLASSIFIER], X_test, y_test, rf_model)
accuracy_reduced = evaluate_classifier_performance(X_train_PCA[:SAMPLES_TO_TRAIN_THE_CLASSIFIER],
                                           y_train[:SAMPLES_TO_TRAIN_THE_CLASSIFIER],
                                                   pca_reduction(X_test, min_components)[0], y_test, rf_model)
print(f"\tAccuracy of the RF classifier on the original dataset: {accuracy_original:.4f}.")
print(f"\tAccuracy of the RF classifier on the PCA reduced dataset: {accuracy_reduced:.4f}.")

# QUESTIONS:
# 9) Is there an important difference between the accuracy of the LR classifier on the original dataset
# and the reduced one?
# Answer: Yes.
# 10) Why?
# Answer: The reduced dataset has fewer dimensions, which means that some information is lost during the PCA reduction.
#         In the original visualization, we showed that the classification problem is not linear.
#         That was illustrated by the fact that t-SNE provided better separation than PCA.
#         PCA performs a linear transformation, which may not be the best choice for this dataset to be classified.

# Let's try it out with t-SNE with just three dimensions and the same classifier

max_number_of_components_t_sne = 3  # t-SNE only allows up to 3 components
X_train_t_sne, t_sne_model = tsne_reduction(X_train[:SAMPLES_TO_TRAIN_THE_CLASSIFIER], max_number_of_components_t_sne)
print(f"Building and evaluating a RF classifier model with t-SNE (just {max_number_of_components_t_sne} dimensions)...")
accuracy_reduced = evaluate_classifier_performance(X_train_t_sne,
                                                   y_train[:SAMPLES_TO_TRAIN_THE_CLASSIFIER],
                                                   t_sne_model.fit_transform(X_test), y_test, rf_model)
print(f"\tAccuracy of the RF classifier on the original dataset: {accuracy_original:.4f}.")
print(f"\tAccuracy of the RF classifier on the t-SNE reduced dataset: {accuracy_reduced:.4f}.")
