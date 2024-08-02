import os
import tensorflow as tf
import numpy as np
from cnn import class_names
from tensorflow.keras.preprocessing import image
from tensorflow import keras
import cropper

# Load the model
model = tf.keras.models.load_model('./nine_layer_cnn/sota2.keras')
img_height = 150
img_width = 150
certainty_factor = .5
labels = ['Clementine', 'Fuji', 'Granny smith', 'Honeycrisp', 'Mandarins',
              'Navel', 'Red Delicious', 'Roma', 'Tomato Cherry Red']

def preprocess_image(image_path:str):
    img = image.load_img(image_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array

def predict(img_array):
    predictions = model.predict(img_array)
    max_index = np.argmax(predictions[0])

    predicted_label = labels[max_index]
    confidence = predictions[0][max_index]

    print(predicted_label)
    print(f"Confidence: {confidence:.2f}%")
def get_classification(image_path:str):
    img_array = preprocess_image(image_path)
    predict(img_array)

    cropped_paths = cropper.crop_image(image_path, 'tmp/', 'YOLO_weights.pt')
    number_of_crops = len(cropped_paths)
    print(number_of_crops)
    for path in cropped_paths:
        img_array = preprocess_image(path)
        predict(img_array)


get_classification('path')