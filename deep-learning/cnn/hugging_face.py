from transformers import TFAutoModelForImageClassification, AutoFeatureExtractor
import tensorflow as tf
from PIL import Image
import requests
import matplotlib.pyplot as plt

import images

# pip install transformers


# Hugging Face

# Hablar de Hugging Face

## Image classification

# Load feature extractor and model


resnet_model_name = "microsoft/resnet-50"
img_feature_extractor = AutoFeatureExtractor.from_pretrained(resnet_model_name)
resnest_model = TFAutoModelForImageClassification.from_pretrained(resnet_model_name)


for url in images.image_URLs:
    # Preprocess the image
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = img_feature_extractor(images=image, return_tensors="tf")

    # Perform inference with the trained model
    outputs = resnest_model(**inputs)
    # get the probability of each class (softmax) a.k.a. the logits
    logits = outputs.logits
    # get the predicted class (the index of the value with the highest probability)
    predicted_class_idx = tf.argmax(logits, axis=-1).numpy()[0]
    # get the name of the predicted class
    label = resnest_model.config.id2label[predicted_class_idx]

    # display the image with the predicted class
    plt.imshow(image)
    plt.title(label)
    plt.axis("off")
    plt.show()

## Object detection

# Hablar de YOLO


# pip install ultralyticsplus==0.0.23 ultralytics==8.0.21
from ultralyticsplus import YOLO, render_result
import images


# load model
model = YOLO('keremberke/yolov8m-plane-detection')

# set model parameters
model.overrides['conf'] = 0.25  # NMS (Non-Maximum Suppression) confidence threshold (0-1). NMS is a technique used to filter out overlapping bounding boxes.
model.overrides['iou'] = 0.45  # NMS IoU (Intersection over Union) threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic means that the NMS will be applied independently to each class
model.overrides['max_det'] = 1000  # maximum number of detections per image


for image_URL in images.plane_image_URLS:
    # perform inference
    results = model.predict(image_URL)
    image = Image.open(requests.get(image_URL, stream=True).raw)
    plt.imshow(image)
    for box in results[0].boxes:
        # get the coordinates of the bounding box (x1, y1, x2, y2)
        x1, y1, x2, y2 = list(box.xyxy[0].numpy())
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color='red', lw=2)
    plt.axis("off")
    plt.show()

## Image segmentation

# Image segmentation is a computer vision technique that involves dividing an image into multiple segments or regions, making it easier to analyze and process. The goal is to simplify the representation of an image by identifying and isolating objects or areas of interest within it. Each segment, region or pixel is assigned a unique label, allowing for easier interpretation and analysis of the image.

# Image segmentation is used in a variety of applications across different fields, including medical imaging (e.g., organ and tumor detection in RMI images), autonomous vehicles (e.g., object detection and tracking), satellite imaging (e.g., smoke detection in forest fires), and augmented reality (e.g., object recognition and tracking).

# Image segmentation can be performed using a variety of techniques, including CNNs (Convolutional Neural Networks).
# In this example, we will use a model for image segmentation using the [Segformer architecture](https://huggingface.co/docs/transformers/model_doc/segformer).
# The [`mattmdjaga/segformer_b2_clothes`](https://huggingface.co/mattmdjaga/segformer_b2_clothes) model is fine-tuned on
# the [ATR dataset](https://github.com/lemondan/HumanParsing-Dataset) for clothes segmentation but can also be used for human segmentation.
# It detects the following 18 classes and segments the image accordingly: Background, Hat, Hair, Sunglasses, Upper-clothes, Skirt, Pants, Dress, Belt, Left-shoe, Right-shoe, Face, Left-leg, Right-leg, Left-arm, Right-arm, Bag, Scarf.


from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import requests
import torch.nn as nn

import images
import utils

# model_name from Hugging Face
model_name = 'mattmdjaga/segformer_b2_clothes'
# the processor takes an input image and returns a tensor
processor = SegformerImageProcessor.from_pretrained(model_name)
# load the model performs the segmentation
model = AutoModelForSemanticSegmentation.from_pretrained(model_name)

# process each image in the list
for image_url in images.segmentation_image_URLs:
    # get the input features from the image
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = processor(images=image, return_tensors="pt")
    # perform segmentation with the model
    outputs = model(**inputs)
    logits = outputs.logits  # get the logits (probabilities for each class) from the output
    # We take the output of the CNN and upsample it to the original image size.
    # That is, we resize the output to the original image size, because the CNN output is smaller than the input image.
    # We use bilinear interpolation to upsample the logits.
    # Then, we will get, for each pixel, the class with the highest probability.
    upsampled_logits = nn.functional.interpolate(logits, size=image.size[::-1], mode="bilinear", align_corners=False)
    # we take the class with the highest probability for each pixel
    segmentation_prediction = upsampled_logits.argmax(dim=1)[0].numpy()
    # Plot the original image and the segmentation mask
    utils.plot_image_segmentation(segmentation_prediction, image, images.class_labels, images.color_mapping)
