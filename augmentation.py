import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def create_augmentation_generator(image_directory, batch_size=32, target_size=(224, 224)):
    """
    This function creates an ImageDataGenerator with various augmentation techniques.

    Args:
    - image_directory (str): The path to the directory containing images to augment.
    - batch_size (int): The number of images to process in each batch.
    - target_size (tuple): The target size to resize images for the model.

    Returns:
    - train_generator: A generator yielding augmented batches of images and labels.
    """
    # Initialize the ImageDataGenerator with several augmentation techniques
    datagen = ImageDataGenerator(
        rescale=1.0/255.0,               # Rescale images to the [0, 1] range
        rotation_range=40,              # Random rotations between 0-40 degrees
        width_shift_range=0.2,          # Random horizontal shifts
        height_shift_range=0.2,         # Random vertical shifts
        shear_range=0.2,                # Random shearing transformations
        zoom_range=0.2,                 # Random zoom in/out
        horizontal_flip=True,           # Random horizontal flips
        fill_mode='nearest'             # Fill the newly created pixels with the nearest value
    )

    # Create the image generator from the directory
    train_generator = datagen.flow_from_directory(
        image_directory,                # Path to your image directory
        target_size=target_size,        # Resize images to the target size
        batch_size=batch_size,          # Define the batch size
        class_mode='categorical',       # Return the labels as categorical (one-hot encoding)
        shuffle=True                    # Shuffle the data before returning
    )

    return train_generator

if __name__ == "__main__":
    # Set the image directory path (adjust this to your own dataset path)
    image_directory = 'path_to_your_image_directory'  # Replace with the actual path to your images

    # Set the batch size and target image size
    batch_size = 32
    target_size = (224, 224)

    # Create the augmentation generator
    generator = create_augmentation_generator(image_directory, batch_size, target_size)

    # Example: Show some augmented images
    import matplotlib.pyplot as plt

    # Fetch a batch of augmented images
    images, labels = next(generator)

    # Display a few augmented images
    for i in range(5):
        plt.imshow(images[i])  # Display the augmented image
        plt.axis('off')
        plt.title(f"Class: {labels[i]}")
        plt.show()
