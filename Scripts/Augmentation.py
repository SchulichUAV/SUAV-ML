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

def save_image_randomly(image: Image.Image, image_name: str, train_folder: str, test_folder: str, object_type: str) -> None:
    '''
    Saves an image either to the train or test folder randomly.
    '''
    if random.random() < 0.6:  # 60% chance to go to train
        dest_folder = os.path.join(train_folder, object_type)
    else: 
        dest_folder = test_folder

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder) 
        
    dest_path = os.path.join(dest_folder, image_name)
    
    try:
        image.save(dest_path)
        return True
    except Exception as e:
        print(f"Error saving image {image_name}: {e}")
        return False

def split_and_augment(raw_folder: str, train_folder: str, test_folder: str, split_ratio=0.8) -> None:
    '''
    Applies random augmentation and splits the dataset into training and testing sets.
    Ensures at least one image from each subfolder is in both train and test.
    '''
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    total_train_images = 0 
    total_test_images = 0 

    # Traverse through each subfolder (object type) in the raw_folder
    for object_type in os.listdir(raw_folder):
        object_folder_path = os.path.join(raw_folder, object_type)
        
        if os.path.isdir(object_folder_path):
            images = [os.path.join(object_folder_path, img) for img in os.listdir(object_folder_path) if img.endswith('.jpg')]
            random.shuffle(images)

            # Ensure at least one image goes to the test set and one to the train set
            split_index = max(1, int(len(images) * split_ratio)) 

            train_images = images[:split_index]
            test_images = images[split_index:]

            # Process train images
            for image_path in train_images:
                try:
                    image = Image.open(image_path)
                    object_train_folder = os.path.join(train_folder, object_type)
                    if not os.path.exists(object_train_folder):
                        os.makedirs(object_train_folder)

                    shutil.copy(image_path, os.path.join(object_train_folder, os.path.basename(image_path)))
                    total_train_images += 1
                    
                    augmented_image = augment_image(image)
                    if save_image_randomly(augmented_image, f"AUG-{os.path.basename(image_path)}", train_folder, test_folder, object_type):
                        total_train_images += 1
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")

            # Process test images
            for image_path in test_images:
                try:
                    image = Image.open(image_path)
                    shutil.copy(image_path, test_folder)
                    total_test_images += 1
                    
                    augmented_image = augment_image(image)
                    if save_image_randomly(augmented_image, f"AUG-{os.path.basename(image_path)}", train_folder, test_folder, object_type):
                        total_test_images += 1
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")

    # Print total images saved
    print(f"Total images saved in training set: {total_train_images}")
    print(f"Total images saved in testing set: {total_test_images}")

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)  # Get the current script's directory
    raw_folder = os.path.join(base_dir, '../Data/Raw_Data')
    train_folder = os.path.join(base_dir, '../Data/Train_Data')
    test_folder = os.path.join(base_dir, '../Data/Test_Data')
    split_and_augment(raw_folder, train_folder, test_folder)