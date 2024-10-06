'''
Script to augment test images.
Is not included in the ML pipeline, but used for generation of diverse test data.
This script should be ran prior to any training / test of the ML system to obtain all necessary data.
'''
import os
import shutil
import random
from PIL import Image
import numpy as np
from imgaug import augmenters as iaa

def augment_image(image: Image.Image) -> Image.Image:
    '''
    Augments image using a set of randomly applied augmentations such as 
    flipping, rotating, brightness adjustment, scaling, noise, etc.

    Parameters:
    image (PIL.Image.Image): The input image to be augmented.

    Returns:
    PIL.Image.Image: The augmented version of the input image.
    '''
    augmenters = iaa.Sequential([
        iaa.Sometimes(0.5, iaa.Fliplr(1.0)),  # horizontal flips 
        iaa.Sometimes(0.5, iaa.Flipud(1.0)),  # vertical flips 
        iaa.Sometimes(0.5, iaa.Affine(rotate=(-180, 180))),  # rotate 
        iaa.Sometimes(0.5, iaa.Multiply((0.6, 1.5))),  # change brightness 
        iaa.Sometimes(0.5, iaa.Affine(scale=(0.4, 2.2))),  # zoom 
        iaa.Sometimes(0.5, iaa.LinearContrast((0.75, 1.5))),  # Adjust contrast 
        iaa.Sometimes(0.4, iaa.AdditiveGaussianNoise(scale=(10, 60))),  # Gaussian noise 
        iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0.0, 3.0))),  # Gaussian blur 
        iaa.Sometimes(0.2, iaa.Dropout(p=(0.1, 0.38))),  # Set a fraction of pixels to zero 
        iaa.Sometimes(0.1, iaa.Grayscale(alpha=(0.0, 1.0))),  # grayscale 
        iaa.Sometimes(0.3, iaa.ElasticTransformation(alpha=50, sigma=5))  # elastic transformations
    ])
    image_np = np.array(image)
    augmented_image_np = augmenters(image=image_np)
    augmented_image = Image.fromarray(augmented_image_np)
    return augmented_image

def split_and_augment(raw_folder: str, train_folder: str, test_folder: str, split_ratio=0.8) -> None:
    '''
    For each raw image, applies random augmentation and splits the dataset into training and testing sets.

    Parameters:
    raw_folder (str): Directory containing original images.
    train_folder (str): Directory where the training images will be saved.
    test_folder (str): Directory where the testing images will be saved.
    split_ratio (float): The proportion of images to use for training (default is 0.8).

    Returns:
    None
    '''
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    # Collect raw images
    raw_images = [os.path.join(raw_folder, filename) for filename in os.listdir(raw_folder) if filename.endswith('.JPG')]
    random.shuffle(raw_images)
    split_index = int(len(raw_images) * split_ratio)
    train_images = raw_images[:split_index]
    test_images = raw_images[split_index:]
    
    # Process train images
    for image_path in train_images:
        image = Image.open(image_path)
        # Copy original image to train
        shutil.copy(image_path, train_folder)
        # Save augmented version to train
        augmented_image = augment_image(image)
        augmented_image_path = os.path.join(train_folder, f"AUG-{os.path.basename(image_path)}")
        augmented_image.save(augmented_image_path)

    # Process test images
    for image_path in test_images:
        image = Image.open(image_path)
        # Copy original image to test
        shutil.copy(image_path, test_folder)
        # Save augmented version to test
        augmented_image = augment_image(image)
        augmented_image_path = os.path.join(test_folder, f"AUG-{os.path.basename(image_path)}")
        augmented_image.save(augmented_image_path)

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)  # Get the current script's directory
    raw_folder = os.path.join(base_dir, '../Data/Raw_Data')
    train_folder = os.path.join(base_dir, '../Data/Train_Data')
    test_folder = os.path.join(base_dir, '../Data/Test_Data')
    split_and_augment(raw_folder, train_folder, test_folder)