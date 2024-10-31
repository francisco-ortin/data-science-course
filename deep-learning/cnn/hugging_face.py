from transformers import TFAutoModelForImageClassification, AutoFeatureExtractor
import tensorflow as tf
from PIL import Image
import requests
import matplotlib.pyplot as plt

from utils import image_URLs

# pip install transformers


# Hugging Face

# Hablar de Hugging Face

## Image classification

# Load feature extractor and model


'''resnet_model_name = "microsoft/resnet-50"
img_feature_extractor = AutoFeatureExtractor.from_pretrained(resnet_model_name)
resnest_model = TFAutoModelForImageClassification.from_pretrained(resnet_model_name)


for url in image_URLs:
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
'''

## Object detection

# Hablar de YOLO


# pip install ultralyticsplus==0.0.23 ultralytics==8.0.21
from ultralyticsplus import YOLO, render_result
from utils import plane_image_URLS

"""
# load model
model = YOLO('keremberke/yolov8m-plane-detection')

# set model parameters
model.overrides['conf'] = 0.25  # NMS (Non-Maximum Suppression) confidence threshold (0-1). NMS is a technique used to filter out overlapping bounding boxes.
model.overrides['iou'] = 0.45  # NMS IoU (Intersection over Union) threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic means that the NMS will be applied independently to each class
model.overrides['max_det'] = 1000  # maximum number of detections per image


for image_URL in plane_image_URLS:
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
"""



import tensorflow as tf
from transformers import TFSegformerForSemanticSegmentation, SegformerFeatureExtractor
import numpy as np
from PIL import Image
import requests

# Load a pre-trained SegFormer model and feature extractor
model_name = "nvidia/segformer-b0-finetuned-ade-512-512"  # Pre-trained model on ADE20k dataset
model = TFSegformerForSemanticSegmentation.from_pretrained(model_name)
feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)

# Define a function for loading and preprocessing the image
def load_and_preprocess_image(image_url):
    response = requests.get(image_url)
    image = Image.open(response.raw).convert("RGB")
    
    # Preprocess the image
    inputs = feature_extractor(images=image, return_tensors="tf")
    return image, inputs

# Define a function to post-process the model output to create segmentation mask
def post_process_segmentation(outputs, target_size=(512, 512)):
    # Get the predicted segmentation mask
    logits = outputs.logits
    upsampled_logits = tf.image.resize(logits, target_size, method="bilinear")
    predicted_mask = tf.argmax(upsampled_logits, axis=-1)[0]
    return predicted_mask

# Load and preprocess the image
image_url = "https://example.com/path/to/your/image.jpg"
image, inputs = load_and_preprocess_image(image_url)

# Run the model
outputs = model(**inputs)

# Post-process the output to get the segmentation mask
segmentation_mask = post_process_segmentation(outputs, target_size=image.size)

# Display the results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(segmentation_mask, cmap="jet", interpolation="nearest")
plt.title("Segmentation Mask")
plt.show()

