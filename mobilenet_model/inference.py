# Import necessary libraries
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the TensorFlow flowers dataset
dataset, info = tfds.load('tf_flowers', with_info=True, as_supervised=True)

# 2. Get the 'train' split
train_data = dataset['train']

# 3. Split the 'train' dataset into training and validation datasets
# We'll take 20% of the 'train' dataset as validation
validation_size = 0.2
train_data_size = len(list(train_data))
validation_data_size = int(train_data_size * validation_size)

# Shuffle and take the first portion as validation
train_data = train_data.shuffle(1000)
validation_data = train_data.take(validation_data_size)
train_data = train_data.skip(validation_data_size)

# 4. Preprocess the image: resizing and normalizing
def preprocess_image(image, label, target_size=(224, 224)):
    """
    Preprocess the image to be compatible with the MobileNet model.
    """
    # Resize the image to the required size for MobileNet (224x224)
    image = tf.image.resize(image, target_size)
    
    # Normalize the image to [0, 1] range
    image = image / 255.0
    
    return image, label

# 5. Prepare the datasets
train_data = train_data.map(preprocess_image).batch(32)
validation_data = validation_data.map(preprocess_image).batch(32)

# 6. Load the pre-trained model
model = tf.keras.models.load_model('mobilenet_flowers.h5')

# 7. Evaluate the model on the validation dataset
loss, accuracy = model.evaluate(validation_data)

print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")

# 8. Test with a few images from the validation dataset and make predictions
# We'll get a batch of images and labels
for images, labels in validation_data.take(1):  # Take one batch of images
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Display the first few images with their predicted classes
    for i in range(5):
        plt.imshow(images[i])  # Display the image
        plt.title(f"Predicted: {predicted_classes[i]}")
        plt.axis('off')
        plt.show()
