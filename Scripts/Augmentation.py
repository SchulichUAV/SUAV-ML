# Script to augment test images
# Is not included in the ML pipeline, but used for generation of diverse test data
import os
from PIL import Image
import numpy as np
from imgaug import augmenters as iaa

def augment_image(image : Image.Image):
    augmenters = iaa.Sequential([
        iaa.Sometimes(0.5, iaa.Fliplr(1.0)),  # horizontal flips 
        iaa.Sometimes(0.5, iaa.Flipud(1.0)),  # vertical flips 
        iaa.Sometimes(0.5, iaa.Affine(rotate=(-180, 180))),  # rotate 
        iaa.Sometimes(0.5, iaa.Multiply((0.6, 1.5))),  # change brightness 
        iaa.Sometimes(0.5, iaa.Affine(scale=(0.5, 2.2))),  # zoom 
        iaa.Sometimes(0.5, iaa.LinearContrast((0.75, 1.5))),  # Adjust contrast 
        iaa.Sometimes(0.4, iaa.AdditiveGaussianNoise(scale=(10, 60))),  # Gaussian noise 
        iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0.0, 3.0))),  # Gaussian blur 
        iaa.Sometimes(0.2, iaa.Dropout(p=(0.1, 0.3))),  # Set a fraction of pixels to zero 
        iaa.Sometimes(0.1, iaa.Grayscale(alpha=(0.0, 1.0))),  # grayscale 
        iaa.Sometimes(0.3, iaa.ElasticTransformation(alpha=50, sigma=5))  # elastic transformations
    ])

    image_np = np.array(image)
    # Apply augmentations
    augmented_image_np = augmenters(image=image_np)
    augmented_image = Image.fromarray(augmented_image_np)
    return augmented_image

def augment_images_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path)
            # Apply augmentations
            augmented_image = augment_image(image)
            # Save augmented image
            output_path = os.path.join(output_folder, filename)
            augmented_image.save(output_path)

if __name__ == "__main__":
    input_folder = '/Users/dominicgartner/Desktop/SUAV/2025ML/2025ML/Data/Raw_Data'
    output_folder = '/Users/dominicgartner/Desktop/SUAV/2025ML/2025ML/Data/Augmented_Data'
    augment_images_in_folder(input_folder, output_folder)