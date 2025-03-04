import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical

# Dataset paths (change these paths to where your 'train' and 'test' folders are)
train_dir = 'D:/PROGRAMMING/RespiraScan/train/'
test_dir = 'D:/PROGRAMMING/RespiraScan/test/'

# Class labels
class_labels = {'NORMAL': 0, 'PNEUMONIA': 1, 'COVID19': 2}

# Image preprocessing function
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Resize image to 224x224
    image = image / 255.0  # Normalize image to [0, 1]
    return img_to_array(image)

# Load data function
def load_data(directory):
    image_paths = []
    labels = []
    
    for category in ['NORMAL', 'PNEUMONIA', 'COVID19']:
        category_path = os.path.join(directory, category)
        for image_name in os.listdir(category_path):
            image_paths.append(os.path.join(category_path, image_name))
            labels.append(class_labels[category])
    
    return image_paths, labels

# Load training and testing data
train_image_paths, train_labels = load_data(train_dir)
test_image_paths, test_labels = load_data(test_dir)

# Preprocess images
X_train = np.array([preprocess_image(img) for img in train_image_paths])
X_test = np.array([preprocess_image(img) for img in test_image_paths])

# Convert labels to one-hot encoding
y_train = to_categorical(np.array(train_labels), num_classes=3)
y_test = to_categorical(np.array(test_labels), num_classes=3)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    
    Dense(3, activation='softmax')  # Output layer for 3 classes
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

# Save the model
model.save('pneumonia_detection_model.h5')
