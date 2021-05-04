import base64
import io

import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os


app = Flask(__name__)
socketio = SocketIO(app)


@app.route('/', methods=['POST', 'GET'])
def index():
    # home Page
    return render_template('index.html')


@socketio.on('image')
def image(data_image):
    # define empty buffer
    frame_buffer = np.empty((0, *dim, channels))

    # unpack request
    for data in data_image:
        # decode and convert the request in image
        img = Image.open(io.BytesIO(base64.b64decode(data[23:])))
        # converting RGB to BGR (opencv standard)
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        # process the frame
        frame_res = cv2.resize(frame, dim)
        #img_tensor = image.img_to_array(frame_res)
        img_tensor = np.array(frame_res)
        #img_tensor = tf.keras.preprocessing.image.array_to_img(frame_res)
        img_tensor.setflags(write=1)
        img_tensor = np.expand_dims(img_tensor, axis=0)   
        img_tensor = img_tensor.astype(np.float32)      
        img_tensor /= 255.    
        #frame_res = frame_res / 255.0
        
    alphabet = [ 
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K",
        "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W",
        "X", "Y", "Z", "del", "nothing", " "]
    predictions = alphabet[np.argmax(model.predict(img_tensor))]
    emit('response_back', predictions)


if __name__ == '__main__':
    # settings
    dim = (200, 200)
    frames = 10
    channels = 3
    model = load_model("my_model.h5")

    # start application
    socketio.run(app=app)