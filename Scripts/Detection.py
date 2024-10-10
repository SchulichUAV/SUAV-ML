'''
Final combined detection file to be ran at the competition.
This script integrates pre-processing steps, saved models and
coordinate calculations of detected objects.
'''
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Utils')))
from Helper import process_image
from PIL import Image

def detect_objects(images: list[Image.Image]) -> dict[str, list[float]]:
    '''
    Detects objects in an image using the YOLO model.
    Images may be passed in by batches.

    Parameters:
    image (PIL.Image.Image): Image to be processed

    Returns:
    dict[str, list[float]]: Dictionary of detected objects and their center coordinates
    '''

    detections = {}

    for image in images:
        processed_image = process_image(image)
    
