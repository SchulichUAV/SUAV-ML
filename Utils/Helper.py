'''
This file contains utility functions that are used across different scripts
to keep common functions organized and avoid code duplication.
'''
from PIL import Image

# Full list of objects that are available for detection. Dataset labels.
# 4 of these objects are randomly selected for detection in the competition.
OBJECTS = [
    'Mannequin',
    'Car',
    'Motorcycle',
    'Airplane',
    'Bus',
    'Boat',
    'Stop Sign',
    'Bench',
    'Snowboard',
    'Umbrella',
    'Sports Ball',
    'Kite',
    'Baseball Bat',
    'Bed',
    'Handbag',
    'Microwave',
    'Clock',
    'Tennis Racket',
    'Suitcase',
    'Skis'
]

# Universal Image sizes / specs for pre-processing
IMG_X = 30
IMG_Y = 30

def process_image(image: Image.Image) -> Image.Image:
    '''
    Pre-processes image to be compatable with models and optimize processing time

    Sources: 
        - https://cs231n.github.io/neural-networks-2/#datapre

    Parameters:
    image (PIL.Image.Image): Image to be processed

    Returns:
    PIL.Image.Image: The processed version of the input image
    '''
    pass

# NOTE: For testing purposes, will be using the Flask server for this later once implemented with the new GCS.
def update_position(old_lat: float, old_long: float, new_lat: float, new_long: float) -> tuple[float, float]:
    updated_lat = (old_lat + new_lat) / 2
    updated_long = (old_long + new_long) / 2
    return updated_lat, updated_long