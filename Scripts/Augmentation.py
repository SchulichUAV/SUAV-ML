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
    '''
    augmenters = iaa.Sequential([
        iaa.Sometimes(0.5, iaa.Fliplr(1.0)),  # horizontal flips
        iaa.Sometimes(0.5, iaa.Flipud(1.0)),  # vertical flips
        iaa.Sometimes(0.5, iaa.Affine(rotate=(-180, 180))),  # rotate
        iaa.Sometimes(0.5, iaa.Multiply((0.6, 1.5))),  # change brightness
        iaa.Sometimes(0.4, iaa.Affine(scale=(0.4, 1.4))),  # zoom
        iaa.Sometimes(0.5, iaa.LinearContrast((0.75, 1.5))),  # Adjust contrast
        iaa.Sometimes(0.4, iaa.AdditiveGaussianNoise(
            scale=(10, 60))),  # Gaussian noise
        iaa.Sometimes(0.3, iaa.GaussianBlur(
            sigma=(0.0, 2.6))),  # Gaussian blur
        # Set a fraction of pixels to zero
        iaa.Sometimes(0.25, iaa.Dropout(p=(0.1, 0.42))),
        iaa.Sometimes(0.25, iaa.Grayscale(alpha=(0.1, 1.0))),  # grayscale
        iaa.Sometimes(0.3, iaa.ElasticTransformation(
            alpha=50, sigma=5))  # elastic transformations
    ])

    image_np = np.array(image)
    augmented_image_np = augmenters(image=image_np)
    augmented_image = Image.fromarray(augmented_image_np)

    # Ensure the image is in RGB mode (JPEG doesn't support RGBA)
    if augmented_image.mode != 'RGB':
        augmented_image = augmented_image.convert('RGB')

    return augmented_image


def save_image(image: Image.Image, image_name: str, train_folder: str, object_type: str) -> None:
    '''
    Saves an image to the train folder in the correct subfolder.
    '''
    dest_folder = os.path.join(train_folder, object_type)

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    dest_path = os.path.join(dest_folder, image_name)

    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image.save(dest_path)
        return True
    except Exception as e:
        print(f"Error saving image {image_name}: {e}")
        return False


def split_and_augment(raw_folder: str, train_folder: str) -> None:
    '''
    Copies all images from raw_folder to train_folder, applies augmentation, 
    and saves augmented images in the same train_folder while maintaining the folder structure.
    '''
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)

    total_train_images = 0

    # Traverse through each subfolder (object type) in the raw_folder
    for object_type in os.listdir(raw_folder):
        object_folder_path = os.path.join(raw_folder, object_type)

        if os.path.isdir(object_folder_path):
            images = [os.path.join(object_folder_path, img) for img in os.listdir(
                object_folder_path) if img.endswith('.jpg')]
            random.shuffle(images)

            # Process each image in the raw folder
            for image_path in images:
                try:
                    image = Image.open(image_path)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')

                    object_train_folder = os.path.join(
                        train_folder, object_type)
                    if not os.path.exists(object_train_folder):
                        os.makedirs(object_train_folder)

                    # Save the original image to the train folder
                    shutil.copy(image_path, os.path.join(
                        object_train_folder, os.path.basename(image_path)))
                    total_train_images += 1

                    # Create and save augmented image
                    augmented_image = augment_image(image)
                    save_image(
                        augmented_image, f"AUG-{os.path.basename(image_path)}", train_folder, object_type)
                    total_train_images += 1
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")

    # Print total images saved
    print(f"Total images saved in training set: {total_train_images}")


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    raw_folder = os.path.join(base_dir, '../Data/Raw_Data')
    train_folder = os.path.join(base_dir, '../Data/Train_Data')

    split_and_augment(raw_folder, train_folder)
