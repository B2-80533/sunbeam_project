from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras import backend as K
from PIL import Image
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import imageio
import io
import os
import h5py


# Define the custom objects here
class MyCustomLayer(tf.keras.layers.Layer):
    def __init__(self, num_filters, kernel_size, **kwargs):
        super(MyCustomLayer, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        # Add custom code to build the layer
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.kernel_size, self.kernel_size, input_shape[-1], self.num_filters),
                                      initializer='uniform',
                                      trainable=True)

    def call(self, inputs):
        # Add custom code to compute the layer output
        return tf.nn.conv2d(inputs, self.kernel, strides=(1, 1), padding='SAME')


def f1(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


# Register the custom objects
custom_objects = {'MyCustomLayer': MyCustomLayer, 'f1': f1}

# Load the model using custom_object_scope
with custom_object_scope(custom_objects):
    model = load_model('/home/oem/Project/flask/InceptionResNetV2.h5')

label_names = {
    0: "Nucleoplasm",
    1: "Nuclear membrane",
    2: "Nucleoli",
    3: "Nucleoli fibrillar center",
    4: "Nuclear speckles",
    5: "Nuclear bodies",
    6: "Endoplasmic reticulum",
    7: "Golgi apparatus",
    8: "Peroxisomes",
    9: "Endosomes",
    10: "Lysosomes",
    11: "Intermediate filaments",
    12: "Actin filaments",
    13: "Focal adhesion sites",
    14: "Microtubules",
    15: "Microtubule ends",
    16: "Cytokinetic bridge",
    17: "Mitotic spindle",
    18: "Microtubule organizing center",
    19: "Centrosome",
    20: "Lipid droplets",
    21: "Plasma membrane",
    22: "Cell junctions",
    23: "Mitochondria",
    24: "Aggresome",
    25: "Cytosol",
    26: "Cytoplasmic bodies",
    27: "Rods & rings"
}

# Define a function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    return preprocessed_img

# Define a function to predict the protein
def predict_protein(img_path):
    img = preprocess_image(img_path)
    preds = model.predict(img)
    # top_preds = decode_predictions(preds, top=3)[0]
    # protein = top_preds[0][1]
    return preds[0]

def decode_predictions(preds, label_names):
    top_preds = preds.argsort()[-3:][::-1]
    result = [(pred, label_names[pred]) for pred in top_preds]
    return result

# Create Server
app = Flask(__name__)

# @app.route("/", methods=["GET"])
# def root():
#     return render_template("new_index.html")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the uploaded image
        img = request.files["image"]
        img_path = os.path.join("static", img.filename)
        img.save(img_path)

        # Predict the protein
        preds = predict_protein(img_path)
        top_preds = decode_predictions(preds, label_names)
        # protein_index, protein_name = top_preds[0]

        # Delete the temporary image file
        os.remove(img_path)

        # Return the prediction
        top_3_proteins = [(pred[1], pred[0]) for pred in top_preds[:3]]
        return f"The top 3 predicted proteins are: {', '.join('{} ({})'.format(p[0], p[1]) for p in top_3_proteins)}"

    return render_template("new_index.html")


# start the server
if __name__ == '__main__':
    app.run(debug=True)
