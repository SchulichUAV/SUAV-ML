'''
This file contains utility functions that are used across different scripts
to keep common functions organized and avoid code duplication.
'''
from PIL import Image

# Full list of objects that are available for detection. Dataset labels
OBJECTS = [
    'Trash Can',
    'Tire',
    'Bike',
    'Basketball',
    'Cone',
    'Mannequin'
]

# Target objects we need to drop payload on.
# We only update the JSON if detection is one of these objects.
# NOTE: UPDATE prior to competition with up-to-date target objects
TARGET_OBJECTS = [

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

def update_position(old_lat: float, old_long: float, new_lat: float, new_long: float) -> tuple[float, float]:
    updated_lat = (old_lat + new_lat) / 2
    updated_long = (old_long + new_long) / 2
    return updated_lat, updated_long