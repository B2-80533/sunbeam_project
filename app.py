import PIL
from flask import Flask, render_template, request, redirect, url_for
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
import io
import os
import h5py
from flask import jsonify

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
    return preds[0]

def decode_predictions(preds, label_names):
    top_preds = preds.argsort()[-3:][::-1]
    result = [(pred, label_names[pred]) for pred in top_preds]
    return result

# Create Server
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
app.config['ALLOWED_EXTENSIONS'] = {'png'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route("/", methods=["GET", "POST"])
def index():
    error_message = None
    if request.method == "POST":
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            return redirect(url_for("result", image=file.filename))

    return render_template("new_index.html", label_names=label_names)

@app.route("/result", methods=["GET"])
def result():
    image_path = request.args.get("image")
    top_3_proteins = []  # Assume no predictions for now
    if image_path:
        preds = predict_protein(os.path.join(app.config['UPLOAD_FOLDER'], image_path))
        top_preds = decode_predictions(preds, label_names)
        top_3_proteins = [(pred[1], pred[0]) for pred in top_preds[:3]]
    return render_template("result.html", image=image_path, top_3_proteins=top_3_proteins)

if __name__ == '__main__':
    app.run(debug=True)




##### VALIDATION INCLUDED ######

# import PIL
# from flask import Flask, render_template, request, redirect, url_for
# import matplotlib.pyplot as plt
# import cv2
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
# import keras
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.utils import custom_object_scope
# from tensorflow.keras import backend as K
# from PIL import Image
# import matplotlib.image as mpimg
# from imgaug import augmenters as iaa
# import imageio
# import io
# import os
# import h5py
# from flask import jsonify
#
#
# # Define the custom objects here
# class MyCustomLayer(tf.keras.layers.Layer):
#     def __init__(self, num_filters, kernel_size, **kwargs):
#         super(MyCustomLayer, self).__init__(**kwargs)
#         self.num_filters = num_filters
#         self.kernel_size = kernel_size
#
#     def build(self, input_shape):
#         # Add custom code to build the layer
#         self.kernel = self.add_weight(name='kernel',
#                                       shape=(self.kernel_size, self.kernel_size, input_shape[-1], self.num_filters),
#                                       initializer='uniform',
#                                       trainable=True)
#
#     def call(self, inputs):
#         # Add custom code to compute the layer output
#         return tf.nn.conv2d(inputs, self.kernel, strides=(1, 1), padding='SAME')
#
#
# def f1(y_true, y_pred):
#     tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
#     fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
#     fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)
#
#     p = tp / (tp + fp + K.epsilon())
#     r = tp / (tp + fn + K.epsilon())
#
#     f1 = 2 * p * r / (p + r + K.epsilon())
#     f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
#     return K.mean(f1)
#
#
# # Register the custom objects
# custom_objects = {'MyCustomLayer': MyCustomLayer, 'f1': f1}
#
# # Load the model using custom_object_scope
# with custom_object_scope(custom_objects):
#     model = load_model('/home/sunbeam/Desktop/PG-DBDA/Project/sunbeam_project/InceptionResNetV2.h5')
#
# label_names = {
#     0: "Nucleoplasm",
#     1: "Nuclear membrane",
#     2: "Nucleoli",
#     3: "Nucleoli fibrillar center",
#     4: "Nuclear speckles",
#     5: "Nuclear bodies",
#     6: "Endoplasmic reticulum",
#     7: "Golgi apparatus",
#     8: "Peroxisomes",
#     9: "Endosomes",
#     10: "Lysosomes",
#     11: "Intermediate filaments",
#     12: "Actin filaments",
#     13: "Focal adhesion sites",
#     14: "Microtubules",
#     15: "Microtubule ends",
#     16: "Cytokinetic bridge",
#     17: "Mitotic spindle",
#     18: "Microtubule organizing center",
#     19: "Centrosome",
#     20: "Lipid droplets",
#     21: "Plasma membrane",
#     22: "Cell junctions",
#     23: "Mitochondria",
#     24: "Aggresome",
#     25: "Cytosol",
#     26: "Cytoplasmic bodies",
#     27: "Rods & rings"
# }
#
# # Define a function to preprocess the image
# def preprocess_image(img_path):
#     img = image.load_img(img_path, target_size=(299, 299))
#     img_array = image.img_to_array(img)
#     expanded_img_array = np.expand_dims(img_array, axis=0)
#     preprocessed_img = preprocess_input(expanded_img_array)
#     return preprocessed_img
#
# # Define a function to predict the protein
# def predict_protein(img_path):
#     img = preprocess_image(img_path)
#     preds = model.predict(img)
#     # top_preds = decode_predictions(preds, top=3)[0]
#     # protein = top_preds[0][1]
#     return preds[0]
#
# def decode_predictions(preds, label_names):
#     top_preds = preds.argsort()[-3:][::-1]
#     result = [(pred, label_names[pred]) for pred in top_preds]
#     return result
#
# # Create Server
# # @app.route("/", methods=["GET"])
# # def root():
# #     return render_template("new_index.html")
# from flask import Flask, render_template, request, redirect, url_for
# from PIL import Image
#
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static'
# app.config['ALLOWED_EXTENSIONS'] = {'png'}
#
# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
#
#
# from PIL import Image
#
# def validate_image(file):
#     try:
#         # Check if the file is None
#         if file.filename is None:
#             return False, "No file selected"
#
#         # Check if the file is allowed
#         if not allowed_file(file.filename):
#             return False, "Invalid file format"
#
#         # Check if the file is a PNG image
#         _, ext = os.path.splitext(file.filename)
#         if ext.lower() != '.png':
#             return False, "Image format must be .png"
#
#         # Validate the image size and dimensions
#         img = Image.open(file)
#         width, height = img.size
#         if width > 512 or height > 512:
#             return False, "Image dimensions must be less than or equal to 512x512 pixels"
#         if file.tell() > 1274400:
#             return False, "Image size must be less than or equal to 1274400 bytes"
#
#         return True, None
#     except Exception as e:
#         return False, f"Error processing image: {e}"
#
#
#
#
# @app.route("/", methods=["GET", "POST"])
# def index():
#     error_message = None
#     if request.method == "POST":
#         if 'image' not in request.files:
#             return redirect(request.url)
#
#         file = request.files['image']
#
#         if file.filename == '':
#             return redirect(request.url)
#
#         if file and allowed_file(file.filename):
#             is_valid, message = validate_image(file)
#             if is_valid:
#                 file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
#                 return redirect(url_for("result", image=file.filename))
#             else:
#                 error_message = message
#
#     return render_template("new_index.html", error_message=error_message, label_names=label_names)
#
# @app.route("/result", methods=["GET"])
# def result():
#     image_path = request.args.get("image")
#     top_3_proteins = []  # Assume no predictions for now
#     if image_path:
#         preds = predict_protein(os.path.join(app.config['UPLOAD_FOLDER'], image_path))
#         top_preds = decode_predictions(preds, label_names)
#         top_3_proteins = [(pred[1], pred[0]) for pred in top_preds[:3]]
#     return render_template("result.html", image=image_path, top_3_proteins=top_3_proteins)
#
#
#
# if __name__ == '__main__':
#     app.run(debug=True)