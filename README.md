# Data science course

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)
![Jupyter Notebooks](https://img.shields.io/badge/Jupyter-Notebook-orange)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()
<img alt="Code size" src="https://img.shields.io/github/languages/code-size/francisco-ortin/data-science-course">
<img alt="Repo size" src="https://img.shields.io/github/repo-size/francisco-ortin/data-science-course">

This is the material of the data science course delivered by [Francisco Ortin](https://www.reflection.uniovi.es/ortin/) 
at the [University of Oviedo](https://www.uniovi.es).

## Table of contents

The course has the following contents (click on the links to access the supporting material):

1. Introduction to data science.
   - What is dat science?
   - Key components of data science.
   - Applications of data science.
   - Data analysis.
   - Types of data analyses.
2. [Data processing and visualization](data-proc-visual).
   - Introduction.
   - Data processing.
   - [NumPy](data-proc-visual/numpy.ipynb).
   - [Pandas](data-proc-visual/pandas.ipynb).
   - Data visualization.
   - Convey true messages.
   - [Matplotlib and Seaborn](data-proc-visual/visualization.ipynb).
3. [Statistical foundations](statistics).
   - Introduction.
   - Descriptive statistics.
     - [Label and one-hot encoding](statistics/encoding.ipynb).
     - [Central tendency](statistics/central.ipynb).
     - [Dispersion](statistics/dispersion.ipynb).
     - [Scaling and normalization](statistics/scaling_normalization.ipynb).
     - [Treating missing values](statistics/missing.ipynb).
     - [Correlation](statistics/correlation.ipynb).
   - Probability.
     - [Central limit theorem](statistics/central_limit.ipynb).
   - Statistical inference.
     - [Confidence intervals](statistics/confidence_intervals.ipynb).
     - [Hypothesis testing](statistics/hypothesis.ipynb).
4. Machine learning.
    - Introduction to machine learning.
    - Supervised machine learning.
    - Unsupervised machine learning.
    - Models, parameters and hyperparameters.
    - Explainability and interpretability.
5. [Regression](regression).
    - Linear regression.
    - Cost function.
      - [Simple linear regression with Scikit-Learn](regression/linear_regression.ipynb).
    - Gradient descent.
      - [Gradient descent](regression/gradient_descent.ipynb).
      - [Learning rate](regression/learning_rate.ipynb).
      - [Batch, stochastic and mini-batch gradient descent](regression/mini_batch.ipynb).
    - Multiple linear regression.
      - [Multiple linear regression](regression/multiple_linear_regression.ipynb).
    - Polynomial regression.
      - [Polynomial regression](regression/polynomial_regression.ipynb).
    - Bias vs. variance.
      - [Choosing the appropriate model complexity](regression/validation.ipynb).
    - Reducing overfitting.
      - [Regularization](regression/regularization.ipynb).
7. [Classification](classification).
    - Introduction.
    - Logistic regression.
      - [Logistic regression](classification/logistic.ipynb).
    - Performance metrics.
      - [Classification metrics](classification/metrics.ipynb).
    - Decision trees.
      - [Decision trees](classification/decision_tree.ipynb).
    - Ensemble methods.
      - [Ensemble methods and model's performance comparison](classification/ensemble.ipynb).
7. [Unsupervised learning](unsupervised).
   - Introduction.
   - Dimensionality reduction.
     - [PCA and t-SNE](unsupervised/pca.ipynb).
   - Clustering.
     - [K-means](unsupervised/k_means.ipynb).
     - [Customer segmentation](unsupervised/customers.ipynb).
   - Anomaly detection.
     - [Variable analysis, Tukey's fences and isolation forest](unsupervised/anomaly.ipynb).
8. [Deep learning](deep-learning).
   - Introduction.
   - Artificial neural networks.
     - [Perceptron neural network for regression](deep-learning/anns/iris_regression.ipynb).
     - [Perceptron for binary classification](deep-learning/anns/iris_single_classifier.ipynb).
     - [Perceptron for multiclass classification](deep-learning/anns/iris_multiple_classifier.ipynb).
     - [TensorFlow and Keras to estimate any differentiable function from data](deep-learning/anns/function_estimation.ipynb).
   - Multilayer perceptron.
     - [The XOR problem](deep-learning/mlp/xor_perceptron.ipynb).
     - [Multilayer perceptron for the XOR problem](deep-learning/mlp/xor_mlp.ipynb).
   - Backpropagation.
   - Activation functions.
     - [Multiclass image classification](deep-learning/activation/image_classifier.ipynb). 
     - [Early stopping](deep-learning/activation/early_stopping.ipynb). 
     - [Different Keras APIs](deep-learning/activation/regression.ipynb). 
     - [Hyperparameter tunning/optimization](deep-learning/activation/hyperparameter.ipynb). 
   - Convolutional neural networks.
     - [Stacking convolution, padding and dense layers](deep-learning/cnn/cnn_architecture.ipynb). 
     - [Transfer learning and data augmentation](deep-learning/cnn/transfer_learning.ipynb). 
     - [Object detection with multiple outputs (multilabel)](deep-learning/cnn/object_detection.ipynb). 
     - [Pretrained deep models from Hugging Face](deep-learning/cnn/hugging_face.ipynb). 
   - Sequence models with recurrent neural networks.
     - [Simple, multiple and deep RNN layers for time series prediction](deep-learning/rnn/simple_rnn.ipynb). 
     - [LSTM and GRU units](deep-learning/rnn/lstm_gru_old.ipynb).
     - [Language models in NLP](deep-learning/rnn/language_model.ipynb).
     - [Contextual word embeddings](deep-learning/rnn/embeddings.ipynb).
     - [Semantic search](deep-learning/rnn/semantic_search.ipynb).
     - [Sentiment analysis](deep-learning/rnn/sentiment.ipynb).
     - [Training deep models with big datasets](deep-learning/rnn/lazy.ipynb).
     - [Encoder-decoder models for nerunal machine translation](deep-learning/rnn/encoder_decoder.ipynb).


Notice that this website only contains the Jupyter Notebooks of the course. That represents a small part of the course material. For any other information about the course, please contact [Francisco Ortin](https://www.reflection.uniovi.es/ortin/).

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](LICENSE) (CC BY-NC-SA 4.0).

[![License: CC BY-NC-SA 4.0](https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/by-nc-sa.svg)](LICENSE)
