# https://github.com/ageron/handson-ml3/blob/main/14_deep_computer_vision_with_cnns.ipynb

# Transfer learning is a powerful machine learning technique where a model trained on one task is repurposed
# or fine-tuned for another related task. Instead of training a new model from scratch,
# transfer learning leverages the knowledge gained from training on a source domain to
# improve performance on a target domain with limited labeled data.

# We use the [Xception deep CNN model](https://ieeexplore.ieee.org/document/8099678) introduced by
# [François Chollet](https://en.wikipedia.org/wiki/Fran%C3%A7ois_Chollet) (the author or Keras) in the paper
# _"Xception: Deep Learning with Depthwise Separable Convolutions"_ published in 2017. It merges the ideas of
# the GoogLeNet and ResNet models, but it replaces the inception modules with a special type
# of layer called a depth-wise separable convolution layer.
# Xception has been trained with the [ImageNet dataset](https://www.image-net.org/), which contains millions of labeled images
# across thousands of categories.

# We use the [tr_flowers dataset](https://www.tensorflow.org/datasets/catalog/tf_flowers?hl=es) from
# TensorFlow with 5 different classes of flowers: dandelion, daisy, tulips, sunflowers, and roses.

# First, Xception is loaded form disc. The last layers are not included, because they are two specific of
# the ImageNet dataset used to train Xception. Then, we add some classifier layers for our problem,
# training the network freezing the weights of the pretrained layers (Xception layers). Finally, all the
# weights are unfrozen and the network is re-trained.
# The resulting classifier has been built by transferring the knowledge of a previous pretrained model (Xception).

# By using Xception model with its trained parameters (weights) we will create a powerful flower classifier
# only with 3670 instances, leveraging the knowledge gained by Xception in other scenarios.

# Particularly, we will use transfer learning to classify flowers of our dataset.

import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
import tensorflow as tf

# DATA PREPARATION

# We first download the "tr_flowers" dataset from tensor flow, which is dataset of flowers
# It only has a train dataset, so we take 10% for test, 15% for val y 75% for train.
(test_raw_ds, val_raw_ds, train_raw_ds), info = tfds.load("tf_flowers", as_supervised=True, with_info=True,
                                                          shuffle_files=True, split=["train[:10%]", "train[10%:25%]", "train[25%:]"])
# Show information about the database
dataset_size = info.splits["train"].num_examples
class_names = info.features["label"].names
n_classes = info.features["label"].num_classes
print(f"Dataset with {dataset_size} examples and the following {n_classes} classes: {class_names}.")

# Let's show the first 9 flowers in the train dataset together with its class
plt.figure(figsize=(12, 10))
index = 0
for image, label in val_raw_ds.take(9):
    index += 1
    plt.subplot(3, 3, index)
    plt.imshow(image)
    plt.title(f"Class: {class_names[label]}")
    plt.axis("off")
plt.show()


# We use keras to resize the images to 224x224 (size of Xception model)
# and rescale the input values to be between -1 and 1
# We can use a feed-forward layer to preprocess the images before feeding them to the model because you will benefit from the GPU acceleration (if any).
preprocess = tf.keras.Sequential([
    # resizes the images to 224x224 (size fo Xception model)
    # aspect ratio is maintained, cropping the file, to not distortion the image
    tf.keras.layers.Resizing(height=224, width=224, crop_to_aspect_ratio=True),
    # rescales the input values to be between -1 and 1
    tf.keras.layers.Lambda(tf.keras.applications.xception.preprocess_input)
])
# resizes and rescales the input
train_ds = train_raw_ds.map(lambda X, y: (preprocess(X), y))

# Shuffles the train dataset.
# The `batch` method collects a number of instances and groups them into a single batch. This is useful divide
# the dataset into mini-batches so that the GPU can process them in parallel.
# prefetch(1) prefetches one data batch at a time from the dataset iterator, ensuring that there is always one
# batch ready to be processed by the model while the current batch is being trained. This overlapping of data
# loading and model execution helps reduce the time spent waiting for data during training,
# thereby improving overall training performance.
# we define the size of the mini-batch
BATCH_SIZE = 32
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE)#.prefetch(1)
val_ds = val_raw_ds.map(lambda X, y: (preprocess(X), y)).batch(BATCH_SIZE)
test_ds = test_raw_ds.map(lambda X, y: (preprocess(X), y)).batch(BATCH_SIZE)


# Let's plot the first 9 images with the new size
plt.figure(figsize=(12, 12))
for X_batch, y_batch in val_ds.take(1):
    for index in range(9):
        plt.subplot(3, 3, index + 1)
        plt.imshow((X_batch[index] + 1) / 2)  # rescale to 0–1 for imshow()
        plt.title(f"Class: {class_names[y_batch[index]]}")
        plt.axis("off")
plt.show()


# MODEL CONSTRUCTION
# We take the Xception model. include_top=False indicates that the last classifier (dense) layers are not included
# The weights (parameters) are those obtained after training the model with the ImageNet database.
base_model = tf.keras.applications.xception.Xception(include_top=False, weights="imagenet")
# The last layer of the Xception model is taken with base_model.output
# Global average pooling computes the mean of the entire feature map
# (it’s like an average pooling layer using a pooling kernel with the same spatial dimensions as the inputs)
# GlobalPooling2D takes a shape of (32, 224, 224, 2048) and converts it to (32, 2048)
pooling_layer = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output_layer = tf.keras.layers.Dense(n_classes, activation="softmax")(pooling_layer)
model = tf.keras.Model(inputs=base_model.input, outputs=output_layer)
# Show the whole model
model.summary()


# TRAINING

# First, we freeze the weights of the pretrained layers
# Only the parameters of the dense layer are tweaked for this classification problem
for layer in base_model.layers:
    layer.trainable = False
# Now, we train the model for three epochs
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(train_ds, validation_data=val_ds, epochs=3)

# val accuracy 0.8657 just trained with 2753 samples to classify 5 categories

# Now, we unfreeze the second half of layers in the base model and retrain for 10 epochs
for layer in base_model.layers[-len(base_model.layers)//2:]:
    layer.trainable = True

early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=[early_stopping_callback])

# Val accuracy grows to 0.9002!

# MODEL EVALUATION

evaluation_results = model.evaluate(test_ds)
print(f"Test loss: {evaluation_results[0]:.4f}. Test accuracy: {evaluation_results[1]:.4f}.")

# Test accuracy is 0.9101 with just a few epochs :-)

# DATA AUGMENTATION

# Data augmentation is a technique commonly used in computer vision to artificially increase the size and diversity
# of a dataset by applying various transformations to the original images. Typical transformations are image flipping,
# rotating and contrasting (among others).

# The following `augment_dataset` takes a dataset and duplicates it with random transformations.

def augment_dataset(dataset):
    """Duplicates the dataset with augmented data (randomly flipped, rotated and contrasted)"""
    data_augmentation_layer = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(mode="horizontal"),
        tf.keras.layers.RandomRotation(factor=0.05),
        tf.keras.layers.RandomContrast(factor=0.2)
    ])
    augmented_ds = dataset.map(lambda X, y: (data_augmentation_layer(X), y))
    # Concatenates the original and augmented dataset (then it shuffles the result)
    return dataset.concatenate(augmented_ds)


# Questions:
# 1) Out of the three datasets (train, val and test), which one do you think would be better augmented?
# Answer: train.
# 2) Why?
# Answer: Because it makes the model to be more efficient (tweak its parameters).
# 3) Modify the example to perform the expected data augmentation.
# Answer: It must be done by moving the following code before the "Model construction" section:
'''
def augment_dataset(dataset):
    """Duplicates the dataset with augmented data (randomly flipped, rotated and contrasted)"""
    data_augmentation_layer = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(mode="horizontal"),
        tf.keras.layers.RandomRotation(factor=0.05),
        tf.keras.layers.RandomContrast(factor=0.2)
    ])
    augmented_ds = dataset.map(lambda X, y: (data_augmentation_layer(X), y))
    # Concatenates the original and augmented dataset (then it shuffles the result)
    return dataset.concatenate(augmented_ds)

train_ds = augment_dataset(train_ds)
train_ds = augment_dataset(train_ds)
'''
# 4) If you have sufficient computation power, re-execute the program with 4 times the number of
#    instances, write the new accuracies obtained and check if you have improved the model.
# Answer:
# In the first training step, val accuracy grows from 0.8657 to 0.8802
# In the second training step, val accuracy grows from 0.9002 to 0.9401
# Test accuracy grows from 0.9101 to 0.9401
