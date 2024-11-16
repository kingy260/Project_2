import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling, RandomZoom, RandomFlip
import matplotlib.pyplot as plt

# Data directories
train_data_dir = "./Project 2 Data/Data/train" 
val_data_dir = "./Project 2 Data/Data/valid" 
test_data_dir = "./Project 2 Data/Data/test" 


# defined image shape
im_shape = (256,256);

# load training and validation images
# Load training images from the directory
train_data = tf.keras.utils.image_dataset_from_directory(
    train_data_dir,              # Directory containing your images
    labels='inferred',         # Infer labels from the directory structure
    label_mode='categorical',          # Use 'int' for integer labels, 'categorical' for one-hot, etc.
    batch_size=32,             # Number of images per batch
    image_size = im_shape,     # Resize images to im_shape pixels
    shuffle=True,               # Shuffle data at the beginning of each epoch
    color_mode='grayscale'
)

# Load validation images
validation_data = tf.keras.utils.image_dataset_from_directory(
    val_data_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size = im_shape,
    shuffle=True,
    color_mode='grayscale'
)

# Load test images
test_data = tf.keras.utils.image_dataset_from_directory(
    test_data_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size = im_shape,
    shuffle=True,
    color_mode='grayscale'
)


for image, label in train_data.take(1):
    print(label.shape)  # Check the shape of the augmented images
    
# augment the training data with a rescale [0,1], random 15% shear slant, random 20% zoom
data_augmentation = Sequential([
    layers.Rescaling(1./255),             
    layers.RandomFlip(mode="horizontal_and_vertical"),     
    layers.RandomZoom(height_factor=0.2, width_factor=0.2)
])

train_data = train_data.map(lambda x, y: (data_augmentation(x), y))

# rescale only for validation data and test
validation_data = validation_data.map(lambda x, y: (tf.keras.layers.Rescaling(1./255)(x), y))

test_data = test_data.map(lambda x, y: (tf.keras.layers.Rescaling(1./255)(x), y))

#Set up the CNN
#Sequential model
model_complex = Sequential()
# Convolutional Layer with 64 filters
model_complex.add(Conv2D(64, (3, 3), activation='relu', input_shape=(256, 256, 1)))
# 2X2 max pooling layer
model_complex.add(MaxPooling2D(pool_size=(2, 2)))
# Second Convolutional Layer with 64 filters, 3x3 kernel, ReLU acitvation function
model_complex.add(Conv2D(64, (3, 3), activation='relu'))
# 2X2 max pooling layer
model_complex.add(MaxPooling2D(pool_size=(2, 2)))
# Flatten the output
model_complex.add(Flatten())
# Fully Connected Layer with 256 units
model_complex.add(Dense(256, activation='relu'))
# Output Layer with 3 classes (crack, missing head, paint off)
model_complex.add(Dense(3, activation='softmax'))
# Compile the model
model_complex.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Display the model's architecture
history = model_complex.summary()

for image, label in train_data.take(1):
    print(image.shape)  # Check the shape of the augmented images

for image, label in train_data.take(1):
    print(label.shape)  # Should be (batch_size, 3) for one-hot labels


model_complex.fit(
    train_data,               # Training data
    epochs=10,                # Number of training epochs
    validation_data=validation_data  # Separate validation set
)

# Evaluate on test data
test_loss, test_acc = model_complex.evaluate(test_data)
print(f'Test accuracy: {test_acc:.4f}')

# Plot training & validation accuracy
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()






