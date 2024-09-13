# 2025-ML

This repo contains the object detection and classification algorithms developed to identify, classify, and determine the spatial coordinates of target objects.

# Specs

- **Framework**: All models are written using **Tensorflow**.

- **Model Architecture**: utilizes three separate CNN models:
  - **Shape Classifier**: Determines the shape of the object within the proposed ROI.
  - **Color Classifier**: Identifies the dominant color of the object within the proposed ROI. The model incorporates a tolerance mechanism to adapt to different illumination levels in different environments.
  - **Letter Classifier**: Recognizes alphanumeric characters contained within the detected object. Model is based on a CNN optimized for text recognition.

<br>

- **Object Detection Backbone**:
  - **YOLOv7**: Real-time object detection. Achieves very high processing speed.

# WorkFlow

![WorkFlow](Doc-Resources/WorkFlow.png)
