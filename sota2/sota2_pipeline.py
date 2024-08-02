import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import cropper

# Load the model
whole_image_model = tf.keras.models.load_model('./nine_layer_cnn/sota2.keras')
crop_image_model = tf.keras.models.load_model('./nine_layer_cnn/sota2_crop.keras')
img_height = 128
img_width = 128
certainty_factor = .5

labels = [
    'Clementine',
    'Fuji',
    'Granny smith',
    'Honeycrisp',
    'Mandarins',
    'Navel',
    'Red Delicious',
    'Roma',
    'Tomato Cherry Red'
]

def preprocess_image(image_path:str):
    img = image.load_img(image_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array


def predict(model, img_array):
    predictions = model.predict(img_array)
    max_index = np.argmax(predictions[0])

    predicted_label = labels[max_index]
    confidence = predictions[0][max_index]

    return predicted_label, confidence


def get_classification(image_path:str):
    labels_dict = {
        labels[0]: 0.0,
        labels[1]: 0.0,
        labels[2]: 0.0,
        labels[3]: 0.0,
        labels[4]: 0.0,
        labels[5]: 0.0,
        labels[6]: 0.0,
        labels[7]: 0.0,
        labels[8]: 0.0
    }

    img_array = preprocess_image(image_path)
    result = predict(whole_image_model, img_array)
    labels_dict[result[0]] += (result[1] * certainty_factor)

    cropped_paths = cropper.crop_image(image_path, './tmp/', './../models/YOLO_weights.pt')
    number_of_crops = len(cropped_paths)
    for path in cropped_paths:
        img_array = preprocess_image(path)
        result = predict(crop_image_model, img_array)
        labels_dict[result[0]] += (result[1] * (certainty_factor/number_of_crops))

    max_label = max(labels_dict, key=labels_dict.get)
    return max_label, labels_dict[max_label]

# # Example usage
# print(get_classification('/path/to/image.jpg'))
