import tensorflow as tf
from tensorflow import keras
from keras.models import load_model, Model
from keras.preprocessing import image
import cv2
import numpy as np
import os
import sys
import time
import cv2

from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from werkzeug.utils import secure_filename
from keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['RESULT_FOLDER'] = 'results/'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg','JPG'])


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

def prepare_image(file):
    # img_path = r''
    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_norm = image.img_to_array(img).astype(np.float32)/255
    img_array_expanded_dims = np.expand_dims(img_norm, axis=0)
    return img_array_expanded_dims

def prediction(filepath):
    model = load_model('LeafBlast_26.hdf5')
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    inp = image.load_img(filepath)
    img_array = image.img_to_array(inp)
    img = prepare_image(filepath)
    classes = model.predict(img)

    c1 = (model.get_layer('conv1'))(img)
    conv1 = tf.Variable(c1)

    bn = (model.get_layer('conv1_bn'))(c1)
    bnorm = tf.Variable(bn)
    rl = (model.get_layer('conv1_relu'))(bn)
    relu = tf.Variable(rl)
    c_dw_1 = (model.get_layer('conv_dw_1'))(rl)
    conv1_dw = tf.Variable(c_dw_1)

    c_dw_1bn = (model.get_layer('conv_dw_1_bn'))(c_dw_1)
    conv1_dw_bn = tf.Variable(c_dw_1bn)
    c_dw_1rl = (model.get_layer('conv_dw_1_relu'))(c_dw_1bn)
    conv1_dw_rl = tf.Variable(c_dw_1rl)
    c_pw_1 = (mobile.get_layer('conv_pw_1'))(c_dw_1rl)
    conv1_pw = tf.Variable(c_pw_1)
    c_pw_1bn = (mobile.get_layer('conv_pw_1_bn'))(c_pw_1)
    conv1_pw_bn = tf.Variable(c_pw_1bn)
    c_pw_1rl = (mobile.get_layer('conv_pw_1_relu'))(c_pw_1bn)
    conv1_pw_rl = tf.Variable(c_pw_1rl)

    res = []   
    if classes[0][0] > classes[0][1]:
        res.append(("Bukan Penyakit Blas",classes[0][0],img_array,img,relu,conv1_pw_rl))
        # print("Daun Padi Sehat")
    elif classes[0][1] > classes[0][0]:
        res.append(("Penyakit Blas Daun Padi",classes[0][1],img_array,img,relu,conv1_pw_rl))
        # print("Penyakit Blas Daun Padi")
    return res

@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        if 'file' in request.files:
            # file = request.files.get('file')
            f = request.files['file']
            if f.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if f and allowed_file(f.filename):
                filename = secure_filename(f.filename)
                # f.save(os.path.join(
                #     app.config['UPLOAD_FOLDER'], filename))
                basepath = os.path.dirname(__file__)
                file_path = os.path.join(
                    basepath, 'static/uploads', secure_filename(f.filename))
                f.save(file_path)
                # return jsonify(predict=prediction(file_path))
                # return prediction(file_path)
                pred = prediction(file_path)
                # return jsonify('success')
                return jsonify(res=str(pred[0][0]), acc=str(pred[0][1]), cit=str(pred[0][2]), proc=str(pred[0][3]), conv=str(pred[0][4]), dept1=str(pred[0][5]))
        else:
            flash('No file part')
            return redirect(request.url)


if __name__ == "__main__":
    app.run(debug=True)
