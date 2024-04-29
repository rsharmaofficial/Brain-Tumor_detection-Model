import cv2
import numpy as np
from PIL import Image

from keras.models import load_model

# Load the model
model = load_model('Braintumour10epochs.keras')

# Read the image
image = cv2.imread('C:\\Users\\risha\\Downloads\\archive (1)\\pred\\pred0.jpg')

# Resize the image
img = Image.fromarray(image)
img = img.resize((64, 64))

# Convert the image to numpy array
img = np.array(img)

# Expand dimensions to match model input shape
input_img = np.expand_dims(img, axis=0)

# Predict the probability scores for each class
result = model.predict(input_img)

# Threshold the probability scores to get the predicted class
predicted_class = 1 if result[0][0] > 0.5 else 0

print(predicted_class)  # Output should be 1 if it's a brain tumor, 0 otherwise
