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

    #matriks citra asli
    inp = image.load_img(filepath)
    img_array = image.img_to_array(inp)

    #matriks pre-processing
    img = prepare_image(filepath)

    #matriks konvolusi standar 3x3
    c1 = (model.get_layer('conv1'))(img)
    conv1 = tf.Variable(c1)
    bn = (model.get_layer('conv1_bn'))(c1)
    bnorm = tf.Variable(bn)
    rl = (model.get_layer('conv1_relu'))(bn)
    relu = tf.Variable(rl)
    c_dw_1 = (model.get_layer('conv_dw_1'))(rl)
    conv1_dw = tf.Variable(c_dw_1)

    #matriks depthwise separable convolution 1
    c_dw_1bn = (model.get_layer('conv_dw_1_bn'))(c_dw_1)
    conv1_dw_bn = tf.Variable(c_dw_1bn)
    c_dw_1rl = (model.get_layer('conv_dw_1_relu'))(c_dw_1bn)
    conv1_dw_rl = tf.Variable(c_dw_1rl)
    c_pw_1 = (model.get_layer('conv_pw_1'))(c_dw_1rl)
    conv1_pw = tf.Variable(c_pw_1)
    c_pw_1bn = (model.get_layer('conv_pw_1_bn'))(c_pw_1)
    conv1_pw_bn = tf.Variable(c_pw_1bn)
    c_pw_1rl = (model.get_layer('conv_pw_1_relu'))(c_pw_1bn)
    conv1_pw_rl = tf.Variable(c_pw_1rl)

    #penambahan operasi padding same/zero padding
    c_pad_2 = (model.get_layer('conv_pad_2'))(c_pw_1rl)
    conv2_pad = tf.Variable(c_pad_2)

    #matriks depthwise separable convolution 2
    c_dw_2 = (model.get_layer('conv_dw_2'))(c_pad_2)
    conv2_dw = tf.Variable(c_dw_2)
    c_dw_2bn = (model.get_layer('conv_dw_2_bn'))(c_dw_2)
    conv2_dw_bn = tf.Variable(c_dw_2bn)
    c_dw_2rl = (model.get_layer('conv_dw_2_relu'))(c_dw_2bn)
    conv2_dw_rl = tf.Variable(c_dw_2rl)
    c_pw_2 = (model.get_layer('conv_pw_2'))(c_dw_2rl)
    conv2_pw = tf.Variable(c_pw_2)
    c_pw_2bn = (model.get_layer('conv_pw_2_bn'))(c_pw_2)
    conv2_pw_bn = tf.Variable(c_pw_2bn)
    c_pw_2rl = (model.get_layer('conv_pw_2_relu'))(c_pw_2bn)
    conv2_pw_rl = tf.Variable(c_pw_2rl)

    #matriks depthwise separable convolution 3
    c_dw_3 = (model.get_layer('conv_dw_3'))(c_pw_2rl)
    conv3_dw = tf.Variable(c_dw_3)
    c_dw_3bn = (model.get_layer('conv_dw_3_bn'))(c_dw_3)
    conv3_dw_bn = tf.Variable(c_dw_3bn)
    c_dw_3rl = (model.get_layer('conv_dw_3_relu'))(c_dw_3bn)
    conv3_dw_rl = tf.Variable(c_dw_3rl) 
    c_pw_3 = (model.get_layer('conv_pw_3'))(c_dw_3rl)
    conv3_pw = tf.Variable(c_pw_3)
    c_pw_3bn = (model.get_layer('conv_pw_3_bn'))(c_pw_3)
    conv3_pw_bn = tf.Variable(c_pw_3bn)
    c_pw_3rl = (model.get_layer('conv_pw_3_relu'))(c_pw_3bn)
    conv3_pw_rl = tf.Variable(c_pw_3rl)

    #matriks depthwise separable convolution 4
    c_pad_4 = (model.get_layer('conv_pad_4'))(c_pw_3rl)
    conv4_pad = tf.Variable(c_pad_4)
    c_dw_4 = (model.get_layer('conv_dw_4'))(c_pad_4)
    conv4_dw = tf.Variable(c_dw_4)
    c_dw_4bn = (model.get_layer('conv_dw_4_bn'))(c_dw_4)
    conv4_dw_bn = tf.Variable(c_dw_4bn)
    c_dw_4rl = (model.get_layer('conv_dw_4_relu'))(c_dw_4bn)
    conv4_dw_rl = tf.Variable(c_dw_4rl)
    c_pw_4 = (model.get_layer('conv_pw_4'))(c_dw_4rl)
    conv4_pw = tf.Variable(c_pw_4)
    c_pw_4bn = (model.get_layer('conv_pw_4_bn'))(c_pw_4)
    conv4_pw_bn = tf.Variable(c_pw_4bn)
    c_pw_4rl = (model.get_layer('conv_pw_4_relu'))(c_pw_4bn)
    conv4_pw_rl = tf.Variable(c_pw_4rl)

    #matriks depthwise separable convolution 5
    c_dw_5 = (model.get_layer('conv_dw_5'))(c_pw_4rl)
    conv5_dw = tf.Variable(c_dw_5)
    c_dw_5bn = (model.get_layer('conv_dw_5_bn'))(c_dw_5)
    conv5_dw_bn = tf.Variable(c_dw_5bn)
    c_dw_5rl = (model.get_layer('conv_dw_5_relu'))(c_dw_5bn)
    conv5_dw_rl = tf.Variable(c_dw_5rl)
    c_pw_5 = (model.get_layer('conv_pw_5'))(c_dw_5rl)
    conv5_pw = tf.Variable(c_pw_5)
    c_pw_5bn = (model.get_layer('conv_pw_5_bn'))(c_pw_5)
    conv5_pw_bn = tf.Variable(c_pw_5bn)
    c_pw_5rl = (model.get_layer('conv_pw_5_relu'))(c_pw_5bn)
    conv5_pw_rl = tf.Variable(c_pw_5rl)

    #penambahan operasi padding same/zero padding
    c_pad_6 = (model.get_layer('conv_pad_6'))(c_pw_5rl)
    conv6_pad = tf.Variable(c_pad_6)
    
    #matriks depthwise separable convolution 6
    c_dw_6 = (model.get_layer('conv_dw_6'))(c_pad_6)
    conv6_dw = tf.Variable(c_dw_6)
    c_dw_6bn = (model.get_layer('conv_dw_6_bn'))(c_dw_6)
    conv6_dw_bn = tf.Variable(c_dw_6bn)
    c_dw_6rl = (model.get_layer('conv_dw_6_relu'))(c_dw_6bn)
    conv6_dw_rl = tf.Variable(c_dw_6rl)
    c_pw_6 = (model.get_layer('conv_pw_6'))(c_dw_6rl)
    conv6_pw = tf.Variable(c_pw_6)
    c_pw_6bn = (model.get_layer('conv_pw_6_bn'))(c_pw_6)
    conv6_pw_bn = tf.Variable(c_pw_6bn)
    c_pw_6rl = (model.get_layer('conv_pw_6_relu'))(c_pw_6bn)
    conv6_pw_rl = tf.Variable(c_pw_6rl)

    #matriks depthwise separable convolution 7
    c_dw_7 = (model.get_layer('conv_dw_7'))(c_pw_6rl)
    conv7_dw = tf.Variable(c_dw_7)
    c_dw_7bn = (model.get_layer('conv_dw_7_bn'))(c_dw_7)
    conv7_dw_bn = tf.Variable(c_dw_7bn)
    c_dw_7rl = (model.get_layer('conv_dw_7_relu'))(c_dw_7bn)
    conv7_dw_rl = tf.Variable(c_dw_7rl)
    c_pw_7 = (model.get_layer('conv_pw_7'))(c_dw_7rl)
    conv7_pw = tf.Variable(c_pw_7)
    c_pw_7bn = (model.get_layer('conv_pw_7_bn'))(c_pw_7)
    conv7_pw_bn = tf.Variable(c_pw_7bn)
    c_pw_7rl = (model.get_layer('conv_pw_7_relu'))(c_pw_7bn)
    conv7_pw_rl = tf.Variable(c_pw_7rl)

    #matriks depthwise separable convolution 8
    c_dw_8 = (model.get_layer('conv_dw_8'))(c_pw_7rl)
    conv8_dw = tf.Variable(c_dw_8)
    c_dw_8bn = (model.get_layer('conv_dw_8_bn'))(c_dw_8)
    conv8_dw_bn = tf.Variable(c_dw_8bn)
    c_dw_8rl = (model.get_layer('conv_dw_8_relu'))(c_dw_8bn)
    conv8_dw_rl = tf.Variable(c_dw_8rl)
    c_pw_8 = (model.get_layer('conv_pw_8'))(c_dw_8rl)
    conv8_pw = tf.Variable(c_pw_8)
    c_pw_8bn = (model.get_layer('conv_pw_8_bn'))(c_pw_8)
    conv8_pw_bn = tf.Variable(c_pw_8bn)
    c_pw_8rl = (model.get_layer('conv_pw_8_relu'))(c_pw_8bn)
    conv8_pw_rl = tf.Variable(c_pw_8rl)

    #matriks depthwise separable convolution 9
    c_dw_9 = (model.get_layer('conv_dw_9'))(c_pw_8rl)
    conv9_dw = tf.Variable(c_dw_9)
    c_dw_9bn = (model.get_layer('conv_dw_9_bn'))(c_dw_9)
    conv9_dw_bn = tf.Variable(c_dw_9bn)
    c_dw_9rl = (model.get_layer('conv_dw_9_relu'))(c_dw_9bn)
    conv9_dw_rl = tf.Variable(c_dw_9rl)
    c_pw_9 = (model.get_layer('conv_pw_9'))(c_dw_9rl)
    conv9_pw = tf.Variable(c_pw_9)
    c_pw_9bn = (model.get_layer('conv_pw_9_bn'))(c_pw_9)
    conv9_pw_bn = tf.Variable(c_pw_9bn)
    c_pw_9rl = (model.get_layer('conv_pw_9_relu'))(c_pw_9bn)
    conv9_pw_rl = tf.Variable(c_pw_9rl)

    #matriks depthwise separable convolution 10
    c_dw_10 = (model.get_layer('conv_dw_10'))(c_pw_9rl)
    conv10_dw = tf.Variable(c_dw_10)
    c_dw_10bn = (model.get_layer('conv_dw_10_bn'))(c_dw_10)
    conv10_dw_bn = tf.Variable(c_dw_10bn)
    c_dw_10rl = (model.get_layer('conv_dw_10_relu'))(c_dw_10bn)
    conv10_dw_rl = tf.Variable(c_dw_10rl)
    c_pw_10 = (model.get_layer('conv_pw_10'))(c_dw_10rl)
    conv10_pw = tf.Variable(c_pw_10)
    c_pw_10bn = (model.get_layer('conv_pw_10_bn'))(c_pw_10)
    conv10_pw_bn = tf.Variable(c_pw_10bn)
    c_pw_10rl = (model.get_layer('conv_pw_10_relu'))(c_pw_10bn)
    conv10_pw_rl = tf.Variable(c_pw_10rl)

    #matriks depthwise separable convolution 11
    c_dw_11 = (model.get_layer('conv_dw_11'))(c_pw_10rl)
    conv11_dw = tf.Variable(c_dw_11)
    c_dw_11bn = (model.get_layer('conv_dw_11_bn'))(c_dw_11)
    conv11_dw_bn = tf.Variable(c_dw_11bn)
    c_dw_11rl = (model.get_layer('conv_dw_11_relu'))(c_dw_11bn)
    conv11_dw_rl = tf.Variable(c_dw_11rl)
    c_pw_11 = (model.get_layer('conv_pw_11'))(c_dw_11rl)
    conv11_pw = tf.Variable(c_pw_11)
    c_pw_11bn = (model.get_layer('conv_pw_11_bn'))(c_pw_11)
    conv11_pw_bn = tf.Variable(c_pw_11bn)
    c_pw_11rl = (model.get_layer('conv_pw_11_relu'))(c_pw_11bn)
    conv11_pw_rl = tf.Variable(c_pw_11rl)

    #penambahan operasi padding same/zero padding
    c_pad_12 = (model.get_layer('conv_pad_12'))(c_pw_11rl)
    conv12_pad = tf.Variable(c_pad_12)

    #matriks depthwise separable convolution 12
    c_dw_12 = (model.get_layer('conv_dw_12'))(c_pad_12)
    conv12_dw = tf.Variable(c_dw_12)
    c_dw_12bn = (model.get_layer('conv_dw_12_bn'))(c_dw_12)
    conv12_dw_bn = tf.Variable(c_dw_12bn)
    c_dw_12rl = (model.get_layer('conv_dw_12_relu'))(c_dw_12bn)
    conv12_dw_rl = tf.Variable(c_dw_12rl)
    c_pw_12 = (model.get_layer('conv_pw_12'))(c_dw_12rl)
    conv12_pw = tf.Variable(c_pw_12)
    c_pw_12bn = (model.get_layer('conv_pw_12_bn'))(c_pw_12)
    conv12_pw_bn = tf.Variable(c_pw_12bn)
    c_pw_12rl = (model.get_layer('conv_pw_12_relu'))(c_pw_12bn)
    conv12_pw_rl = tf.Variable(c_pw_12rl)

    #matriks depthwise separable convolution 13
    c_dw_13 = (model.get_layer('conv_dw_13'))(c_pw_12rl)
    conv13_dw = tf.Variable(c_dw_13)
    c_dw_13bn = (model.get_layer('conv_dw_13_bn'))(c_dw_13)
    conv13_dw_bn = tf.Variable(c_dw_13bn)
    c_dw_13rl = (model.get_layer('conv_dw_13_relu'))(c_dw_13bn)
    conv13_dw_rl = tf.Variable(c_dw_13rl)
    c_pw_13 = (model.get_layer('conv_pw_13'))(c_dw_13rl)
    conv13_pw = tf.Variable(c_pw_13)
    c_pw_13bn = (model.get_layer('conv_pw_13_bn'))(c_pw_13)
    conv13_pw_bn = tf.Variable(c_pw_13bn)
    c_pw_13rl = (model.get_layer('conv_pw_13_relu'))(c_pw_13bn)
    conv13_pw_rl = tf.Variable(c_pw_13rl)

    #matriks average pooling
    avg_pool = (model.get_layer('global_average_pooling2d_1'))(c_pw_13rl)
    gap = tf.Variable(avg_pool)

    #matriks dropout 0.4
    dpo = (model.get_layer('dropout_1'))(avg_pool)
    dropout = tf.Variable(dpo)

    #matriks fully connected layer dan aktivasi softmax
    den = (model.get_layer('dense_1'))(dpo)
    dense = tf.Variable(den)

    #proses prediksi
    classes = model.predict(img)
    res = []   
    if classes[0][0] > classes[0][1]:
        result = str(classes[0][0] * 100)+'%'
        res.append(("Bukan Penyakit Blas",result,img_array,img,relu,conv1_pw_rl,conv2_pw_rl,conv3_pw_rl,conv4_pw_rl,conv5_pw_rl,conv6_pw_rl,conv7_pw_rl,conv8_pw_rl,conv9_pw_rl,conv10_pw_rl,conv11_pw_rl,conv12_pw_rl,conv13_pw_rl,gap,dropout,dense))
        # print("Daun Padi Sehat")
    elif classes[0][1] > classes[0][0]:
        result = str(classes[0][1] * 100)+'%'
        res.append(("Penyakit Blas Daun Padi",result,img_array,img,relu,conv1_pw_rl,conv2_pw_rl,conv3_pw_rl,conv4_pw_rl,conv5_pw_rl,conv6_pw_rl,conv7_pw_rl,conv8_pw_rl,conv9_pw_rl,conv10_pw_rl,conv11_pw_rl,conv12_pw_rl,conv13_pw_rl,gap,dropout,dense))
        # print("Penyakit Blas Daun Padi")
    return res

@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        if 'file' in request.files:
            f = request.files['file']
            if f.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if f and allowed_file(f.filename):
                filename = secure_filename(f.filename)
                basepath = os.path.dirname(__file__)
                file_path = os.path.join(
                    basepath, 'static/uploads', secure_filename(f.filename))
                f.save(file_path)
                pred = prediction(file_path)
                return jsonify(res=str(pred[0][0]), acc=str(pred[0][1]), cit=str(pred[0][2]), proc=str(pred[0][3]), conv=str(pred[0][4]), dept1=str(pred[0][5]), dept2=str(pred[0][6]), dept3=str(pred[0][7]), dept4=str(pred[0][8]), dept5=str(pred[0][9]), dept6=str(pred[0][10]), dept7=str(pred[0][11]), dept8=str(pred[0][12]), dept9=str(pred[0][13]), dept10=str(pred[0][14]), dept11=str(pred[0][15]), dept12=str(pred[0][16]), dept13=str(pred[0][17]), gap=str(pred[0][18]), drop=str(pred[0][19]), dense=str(pred[0][20]))
        else:
            flash('No file part')
            return redirect(request.url)


if __name__ == "__main__":
    app.run(debug=True)
