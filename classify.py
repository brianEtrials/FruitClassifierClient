import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

def classify_image(filepath, modelpath, labels):
    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    model = tf.keras.models.load_model(modelpath)
    predictions = model.predict(img_array)
    max_index = np.argmax(predictions[0])
    predicted_label = labels[max_index]
    confidence = predictions[0][max_index]
    return predicted_label, confidence
