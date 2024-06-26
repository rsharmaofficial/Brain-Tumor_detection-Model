import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

image_directory = 'dataset/'  # Ensure correct path
dataset = []
label = []

INPUT_SIZE = 64  # Fixing the input size to 64

no_tumor_images = os.listdir(image_directory + 'no')
yes_tumor_images = os.listdir(image_directory + 'yes')

for i, image_name in enumerate(no_tumor_images):
    if image_name.lower().endswith('.jpg'):  # Check for lowercase .jpg extension
        try:
            image = cv2.imread(image_directory + 'no/' + image_name)
            image = Image.fromarray(image, 'RGB')
            image = image.resize((INPUT_SIZE, INPUT_SIZE))
            dataset.append(np.array(image))
            label.append(0)
        except (OSError, cv2.error):  # Handle potential image reading errors
            print(f"Error reading image: {image_name}")

for i, image_name in enumerate(yes_tumor_images):
    if image_name.lower().endswith('.jpg'):  # Check for lowercase .jpg extension
        try:
            image = cv2.imread(image_directory + 'yes/' + image_name)
            image = Image.fromarray(image, 'RGB')
            image = image.resize((INPUT_SIZE, INPUT_SIZE))
            dataset.append(np.array(image))
            label.append(1)
        except (OSError, cv2.error):  # Handle potential image reading errors
            print(f"Error reading image: {image_name}")

# Convert both dataset and label to NumPy arrays (assuming they are lists)
dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

# Normalizing the data
x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

y_train=to_categorical(y_train,num_classes=2)
y_test=to_categorical(y_test,num_classes=2)
# Model building
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('sigmoid'))

# Binary CrossEntropy=1, sigmoid
# Categorical CrossEntropy=2, softmax
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=16,verbose=1,epochs=10,validation_data=(x_test,y_test),shuffle=False)


model.save('Braintumour10epochscategorical.keras')

# Checking the summary of the model
# model.summary()
