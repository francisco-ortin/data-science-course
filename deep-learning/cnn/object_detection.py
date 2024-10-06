# https://www.kaggle.com/code/hhp07022000/object-localization-with-mnist

# OBJECT DETECTION

# CNNs are not only used for image classification. A powerful feature of ANNs is that they can have multiple outputs.
# Thus, a CNN can be used to not only recognize an object in an image but also localize it in the picture.
# This is what is called *object detection*.
# Localizing an object in a picture can be expressed as a regression task. We have to predict a bounding
# box around the object: the horizontal and vertical coordinates of the object’s center (two numbers),
# as well as its height and width (other two). This means we have four numbers to predict (four regression problems).

# This example predicts the MNIST digit in an image and also localizes its position: digit detection.


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras as ks
from keras import Input, Model

import utils


# DATA PREPARATION

BATCH_SIZE = 32

# We get the MNIST dataset (train, val and test) and generate the bounding boxes around the digits
train_dataset, val_dataset = utils.get_train_val_dataset(BATCH_SIZE)
test_dataset = utils.get_test_dataset()

# Converts the raw dataset tensorflow format to numpy (before its visualization)
(train_digits, train_labels, train_bboxes,
 val_digits, val_labels, val_bboxes,
 test_digits, test_labels, test_bboxes) = \
    utils. dataset_to_numpy_util(train_dataset, val_dataset, test_dataset, 10)


# We visualize digits and bounding boxes of train and validation datasets
utils.display_digits_with_boxes(train_digits, train_labels, train_labels, np.array([]),
                                train_bboxes, 'Training digits with their labels and bounding boxes')
utils.display_digits_with_boxes(val_digits, val_labels, val_labels, np.array([]),
                                val_bboxes, 'Validation digits with their labels and bounding boxes')
plt.show()


# MODEL CREATION
# We will use the functional API (not the sequential one) to build the model.
# The reason is that there are different outputs and hence the sequential is not suitable.

def cnn_network(input_layer: Input) -> Model:
    """Creates the CNN network for the specified input.
    CNN and pooling layers. Then a dense layer is added at the end of the network."""
    x = ks.layers.Conv2D(16, activation='relu', kernel_size=3, input_shape=(75, 75, 1))(input_layer)
    x = ks.layers.MaxPooling2D((2, 2))(x)
    x = ks.layers.Conv2D(32, activation='relu', kernel_size=3)(x)
    x = ks.layers.MaxPooling2D((2, 2))(x)
    x = ks.layers.Conv2D(64, activation='relu', kernel_size=3)(x)
    x = ks.layers.MaxPooling2D((2, 2))(x)
    # Takes the CNN network, flattens it to a 1D tensor and adds one dense layer
    x = ks.layers.Flatten()(x)
    x = ks.layers.Dense(128, activation='relu')(x)
    return x


def classifier_output(cnn: Model) -> Model:
    """This is the first output of the network.
    Takes the CNN and adds a softmax dense layer to classify the digit."""
    return ks.layers.Dense(10, activation='softmax', name='classifier')(cnn)


def bounding_box_regression_output(cnn_network: Model) -> Model:
    """This is the second output of the network.
    Takes the CNN and adds 4 regression outputs for the bounding box (rectangle)."""
    bounding_box_regression_output = ks.layers.Dense(units='4', name='bounding_box')(cnn_network)
    return bounding_box_regression_output


def final_model(input_layer) -> Model:
    """Creates the final model with an input layer and two output layers (classification and 4 regressions)."""
    cnn = cnn_network(input_layer)
    classification_output = classifier_output(cnn)
    bounding_box = bounding_box_regression_output(cnn)
    # The final network has two outputs
    return ks.Model(inputs=input_layer, outputs=[classification_output, bounding_box])


def build_and_compile_model(input_layer: Input) -> Model:
    """Builds and compiles the final model"""
    model = final_model(input_layer)
    # This model has two outputs, so two loss functions must be defined.
    # Cross entropy for the classifier and MSE for the 4 regressors.
    losses = {'classifier': 'categorical_crossentropy', 'bounding_box': 'mse'}
    # The same for the metrics.
    metrics = {'classifier': 'acc', 'bounding_box': 'mse'}
    model.compile(optimizer='adam', loss=losses, metrics=metrics)
    return model


# Input layer comprises images of 75x75 pixels in grayscale (one color)
inputs = tf.keras.layers.Input(shape=(75, 75, 1,))
model = build_and_compile_model(inputs)
model.summary()

# MODEL TRAINING

history = model.fit(train_dataset, validation_data=val_dataset, epochs=5)


# MODEL EVALUATION

# We evaluate the model's performance with the test dataset
loss, classification_loss, bounding_box_loss, classification_accuracy, bounding_box_mse = (
    model.evaluate(test_dataset, steps=1))

print(f"Evaluation with the test dataset:\n"
      f"\t - Loss={loss:.4f}.\n"
      f"\t - Classification loss={classification_loss:.4f}.\n"
      f"\t - Bounding box loss={bounding_box_loss:.4f}.\n"
      f"\t - Classification accuracy={classification_accuracy:.4f}.\n"
      f"\t - Bounding box MSE={bounding_box_mse:.4f}.")


# MODEL PREDICTION

# Let's visualize how the model predicts the label of each digit and its location
predictions_classifier_proba, predicted_bounding_boxes = model.predict(test_digits)
prediction_labels = np.argmax(predictions_classifier_proba, axis=1)

utils.display_digits_with_boxes(test_digits, prediction_labels, test_labels,
                                predicted_bounding_boxes, test_bboxes, 'True and predicted values for the test set')

plt.show()


# Questions:
# 1) How many parameters does this network use? Do you think the model has a lot of parameters?
# Answer: 426,638 params. Given that is a CNN for object detection, they are not a lot of them.
# 2) Do you think it has reasonable performance?
# Answer: With those parameters and 5 epochs, it gets Classification accuracy=0.9831 and
#         bounding box MSE=0.0017 which is very good.
# 3) Do you think it could be improved if we add more epochs in the training process?
# Answer: Yes. All the val (and train) measures were growing.
# 4) What could be done if object detection is difficult (e.g., objects are hard to recognize and image background
#    makes it hard to localize the objects) and performance is not the expected one?
# Answer: Two things 1) data augmentation and 2) transfer learning (we have not done it in this example to
#         train the models in a faster way).
