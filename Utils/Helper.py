'''
This file contains utility functions that are used across different scripts
to keep common functions organized and avoid code duplication.
'''
from PIL import Image

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