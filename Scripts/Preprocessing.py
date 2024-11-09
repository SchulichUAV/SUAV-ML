import os
from PIL import Image
import numpy as np

#global vars
target_size = (608,608)

def process_image(image: Image.Image) -> Image.Image:
    global target_size

    #limit the rgb between 0 and 1
    image = normalize_image(image)

    #letter box image to ensure same aspect ratio
    final_image = letter_box(image, target_size)

    return final_image
    

def _normalize_image(image: Image.Image) -> Image.Image:
    """
    Normalizes pixels so that they range between [0, 1]
    """
    #get image array and divide the values by 255
    img_arr = np.array(image, np.float32)

    norm_arr = img_arr/255.0

    norm_arr = Image.fromarray((norm_arr).astype(np.uint8))

    return norm_arr

def _letter_box(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """
    Letterboxes the image to ensure aspect ratio is correct without distorting it.
    Resizes the image to fit within the target size and pastes it onto a black background.
    """
    
    # Calculate aspect ratio
    img_width, img_height = image.size
    target_width, target_height = target_size

    a_r = img_width / img_height

    # Determine new width and height based on aspect ratio
    if a_r > 1:
        new_width = target_width
        new_height = int(target_width / a_r)
    else:
        new_height = target_height
        new_width = int(target_height * a_r)

    # Resize the image while maintaining aspect ratio and paste
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

    new_image = Image.new('RGB', target_size, (0, 0, 0))

    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    new_image.paste(resized_image, (x_offset, y_offset))

    return new_image