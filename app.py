from __future__ import division, print_function
import os
import numpy as np
import tensorflow as tf


from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
import keras.utils as image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

MODEL_PATH = 'soil.h5'
# model = pickle.load(open(MODEL_PATH, 'rb'))
model = load_model(MODEL_PATH)
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = tf.keras.utils.load_img(img_path, target_size=(150, 150))

    x = image.img_to_array(img)
    x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)
    pred = np.argmax(model.predict(x))
    return pred


@app.route('/', methods=['GET'])
def index():
    return "Welcome to the soil app ml backend"


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path, model)
        print(str(preds))

        labels = ['Alluvial_Soil', 'Black_Soil', 'Clay_Soil', 'Red_Soil']
        os.remove(file_path)
        return labels[preds]
    return None


if __name__ == '__main__':
    app.run(debug=True)
