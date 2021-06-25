import tensorflow as tf
from tensorflow import keras
from keras.models import load_model, Model
from keras.preprocessing import image
import cv2
import numpy as np

model = load_model('LeafBlast_26.hdf5')
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

def prepare_image(file):
    # img_path = r''
    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_norm = image.img_to_array(img).astype(np.float32)/255
    img_array_expanded_dims = np.expand_dims(img_norm, axis=0)
    return img_array_expanded_dims

# img = cv2.imread('static/uploads/IMG_5611.JPG')
# img = cv2.resize(img,(224,224))
# img = np.reshape(img,[1,224,224,3])
img = prepare_image('static/uploads/IMG_5617.JPG')
# print(img.shape)
# prediction = mo.predict(preprocessed_image)
# img = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(os.path.join(dir,"img (224).jpg")), target_size=(224, 224), batch_size=(32), class_mode='categorical')
# img = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(os.path.join(base_dir,"test"), target_size=(224, 224), batch_size=(32), class_mode='categorical')
classes = model.predict(img) 
# print(classes)
res = []
if classes[0][0] > classes[0][1]:
    res.append(("Daun Padi Sehat",classes[0][0]))
    # print("Daun Padi Sehat")
elif classes[0][1] > classes[0][0]:
    res.append(("Penyakit Blas Daun Padi",classes[0][1]))
    # print("Penyakit Blas Daun Padi")
print(res)